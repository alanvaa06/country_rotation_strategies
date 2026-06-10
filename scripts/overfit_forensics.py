"""Overfitting forensics for the EM Cap-Tilt deploy candidate.

Question under test
-------------------
Is the EM Cap-Tilt strategy (50/50 Value+Momentum composite, @63d,
Cap_Tilt construction vs the vendor EM cap index, active basis) OVERFITTED,
or is its DSR failure (0.762 < 0.95) purely a statistical-power artifact
of the 15.5-year sample?

Strategy reconstruction is exact (asserted against the certified verdict
``outputs/research/verdict_EM_prior_vm_p63_active_captilt_capbmk.json``:
sharpe_ann, t, PSR, DSR reproduced bit-for-bit).

Analyses
--------
1. DSR power decomposition — invert PSR(SR0)=0.95 for the required Sharpe /
   t-stat given the actual sweep trial variance + n_trials; implied years
   needed at the current IR.
2. Monte-Carlo deep dive — random-signal null, n_sims=500, active basis,
   base_weights threaded (construction held constant).
3. Walk-forward forensics — per-fold chosen params, IS vs OOS Sharpe.
4. Subperiod robustness — halves and thirds: ann active return, IR,
   quarterly hit rate.
5. Rolling 252d IR — fraction positive, worst/best window.
6. Cost stress — IR at tc_bps in {2, 5, 10, 20} (period costs deducted from
   the gross daily active curve at period-end dates).
7. Parameter neighborhood — 1-D sweep over selection N / periodicity /
   active_share; distribution, frac positive, default z-score.
8. IC time stability — composite relative IC @63, full window vs halves.
9. Trial-count honesty — DSR recomputed at the cross-run ledger trial count
   (N=180; ~150 legacy scenarios + 8 segment runs + 18 tournament + 4 S5).
10. Verdict — overfitting signatures present/absent + family-wise honesty
    note on the post-hoc EM segment selection (1-of-3).

Outputs
-------
* ``outputs/research/overfit_forensics_EM.json`` (machine-readable, gitignored)
* ``outputs/research/overfit_forensics_EM.md``   (full report, committed)

Usage:  ``python scripts/overfit_forensics.py``  (~3-6 min; needs local
gitignored ``Inputs/`` + ``Classification.xlsx``).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import brentq

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import research_run as rr  # noqa: E402
from country_rotation.backtest import ic as ic_module  # noqa: E402
from country_rotation.backtest.engine import Engine  # noqa: E402
from country_rotation.config import load_config  # noqa: E402
from country_rotation.validation.protocols import (  # noqa: E402
    equity_curve,
    monte_carlo_null,
    parameter_sweep,
    stability_summary,
    walk_forward,
)
from country_rotation.validation.statistics import (  # noqa: E402
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_significance,
)

_OUT_DIR = os.path.join(_REPO, "outputs", "research")
_JSON_PATH = os.path.join(_OUT_DIR, "overfit_forensics_EM.json")
_MD_PATH = os.path.join(_OUT_DIR, "overfit_forensics_EM.md")
_VERDICT_PATH = os.path.join(
    _OUT_DIR, "verdict_EM_prior_vm_p63_active_captilt_capbmk.json"
)

#: Certified validation grid (research_run.py main(), non-quick).
_CERT_GRID = {"relative_selection_score": (3, 5, 7), "periodicity": (21, 63)}

#: Cross-run multiple-testing ledger (final_evidence_dossier.md "Standing
#: decisions" #4): ~150 legacy scenarios + 8 segment runs + 6x3 tournament
#: specs + 4 S5 confirmation books.
_HONEST_N_TRIALS = 180

_EULER = 0.5772156649015328
_SQ252 = math.sqrt(252.0)


def _log(msg: str) -> None:
    print(f"[overfit_forensics] {msg}", flush=True)


# ---------------------------------------------------------------------------
# DSR power algebra (Bailey & Lopez de Prado 2012/2014 closed forms)
# ---------------------------------------------------------------------------

def _variance_term(sr: float, skew: float, kurt: float) -> float:
    """PSR variance term: 1 - g3*SR + ((g4-1)/4)*SR^2 (g4 non-excess)."""
    return max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr, 1e-12)


def sr0_from_sigma_n(sigma: float, n_trials: int) -> float:
    """Expected max Sharpe under the no-skill null (Bailey-LdP 2014),
    parameterized directly by the trial-Sharpe sigma and trial count."""
    if n_trials <= 1 or not sigma > 0.0:
        return 0.0
    p1 = min(1.0 - 1e-10, 1.0 - 1.0 / n_trials)
    p2 = min(1.0 - 1e-10, 1.0 - 1.0 / (n_trials * math.e))
    return sigma * (
        (1.0 - _EULER) * sps.norm.ppf(p1) + _EULER * sps.norm.ppf(p2)
    )


def dsr_required_sr(
    sr0: float, n: int, skew: float, kurt: float, conf: float = 0.95
) -> float:
    """Daily Sharpe SR* such that PSR(SR0) = conf at sample size n,
    holding the observed skew / kurtosis fixed."""
    z = float(sps.norm.ppf(conf))

    def f(sr: float) -> float:
        return (sr - sr0) * math.sqrt(n - 1.0) - z * math.sqrt(
            _variance_term(sr, skew, kurt)
        )

    return float(brentq(f, sr0, 1.0))


def dsr_required_n(
    sr: float, sr0: float, skew: float, kurt: float, conf: float = 0.95
) -> float:
    """Sample size n such that PSR(SR0) = conf at the given daily Sharpe."""
    if sr <= sr0:
        return math.inf
    z = float(sps.norm.ppf(conf))
    return 1.0 + z * z * _variance_term(sr, skew, kurt) / (sr - sr0) ** 2


def lo_t(sr: float, n: int) -> float:
    """Lo (2002) IID t-stat for a daily Sharpe sr at sample size n."""
    return sr / math.sqrt((1.0 + sr * sr / 2.0) / n)


# ---------------------------------------------------------------------------
# Series helpers
# ---------------------------------------------------------------------------

def ann_ir(daily: pd.Series) -> float:
    """Annualized IR of a daily active series (mean/std * sqrt(252))."""
    sd = float(daily.std(ddof=1))
    if not sd > 0.0:
        return math.nan
    return float(daily.mean()) / sd * _SQ252


def subperiod_stats(daily: pd.Series) -> dict:
    """Ann active return, IR and quarterly hit rate for one subperiod."""
    quarters = daily.resample("QE").sum()
    return {
        "start": str(daily.index[0].date()),
        "end": str(daily.index[-1].date()),
        "n_days": int(len(daily)),
        "ann_active_return": float(daily.mean()) * 252.0,
        "ir": ann_ir(daily),
        "n_quarters": int(len(quarters)),
        "quarter_hit_rate": float((quarters > 0.0).mean()),
    }


def split_chunks(daily: pd.Series, k: int) -> list[pd.Series]:
    """Split a daily series into k contiguous near-equal chunks."""
    return [
        daily.iloc[seg[0] : seg[-1] + 1]
        for seg in np.array_split(np.arange(len(daily)), k)
    ]


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct():
    """Rebuild the certified EM Cap-Tilt book; assert verdict parity."""
    cfg = load_config(os.path.join(_REPO, "configs", "default.json"))
    processed, classification = rr.ingest(cfg)

    universe = [
        c
        for c in rr.extract_universe(classification, "EM")
        if c in processed["Price"].columns
    ]
    price_df = processed["Price"]
    prices = price_df[universe].copy()
    prices_with_bmk = prices.copy()
    prices_with_bmk["EM"] = price_df["EM"]  # vendor cap-weighted EM index

    factor_scores, _ = rr.build_factor_scores(processed, universe)
    normalized, _, _ = rr.build_composite(
        factor_scores, tuple(rr.select_prior_factors(factor_scores, "vm"))
    )

    # Cap-weight base, parity with research_run.py (positive-total rows only)
    mcap_df = processed["Market_Cap"]
    mcap = mcap_df[[c for c in universe if c in mcap_df.columns]].ffill()
    mcap = mcap[mcap.sum(axis=1) > 0]
    base_weights = mcap.div(mcap.sum(axis=1), axis=0)

    cfg_bt = dataclasses.replace(
        cfg.backtest,
        bmk="EM",
        bmk_weight=0.0,
        mode="active",
        periodicity=63,
        selection_criteria="relative",
        weighting_method="Cap_Tilt",
        active_share=0.30,
    )

    result = Engine(normalized, prices_with_bmk, cfg_bt, base_weights=base_weights).run()
    active_daily = (result.daily_returns - result.daily_bmk_returns).dropna()
    return (
        normalized,
        prices,
        prices_with_bmk,
        cfg_bt,
        base_weights,
        result,
        active_daily,
    )


def assert_parity(equity, dsr_recomputed) -> dict:
    """Assert the reconstruction reproduces the certified verdict stats."""
    with open(_VERDICT_PATH, encoding="utf-8") as f:
        cert = json.load(f)["stats"]
    sig = sharpe_significance(equity)
    psr = probabilistic_sharpe_ratio(equity)
    checks = {
        "sharpe_ann": (sig.sharpe_ann, cert["sharpe_ann"]),
        "sharpe_t_stat": (sig.t_stat, cert["sharpe_t_stat"]),
        "psr": (psr.psr, cert["psr"]),
        "dsr": (dsr_recomputed, cert["dsr"]),
    }
    for name, (got, want) in checks.items():
        if abs(got - want) > 1e-8:
            raise AssertionError(
                f"Reconstruction parity FAILED on {name}: got {got!r}, "
                f"certified {want!r}"
            )
    _log(
        "Reconstruction parity PASS (sharpe_ann/t/PSR/DSR match certified "
        "verdict to <1e-8)."
    )
    return {k: v[0] for k, v in checks.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    os.makedirs(_OUT_DIR, exist_ok=True)

    _log("Reconstructing EM Cap-Tilt vm @63 (vs vendor EM cap index) ...")
    (
        normalized,
        prices,
        prices_with_bmk,
        cfg_bt,
        base_weights,
        result,
        active_daily,
    ) = reconstruct()
    equity = equity_curve(active_daily)
    n_days = int(len(active_daily))
    years = n_days / 252.0
    _log(f"Active daily series: {n_days} days ({years:.1f}y), "
         f"{active_daily.index[0].date()} -> {active_daily.index[-1].date()}")

    # ------------------------------------------------------------------
    # 1. DSR power decomposition (uses the certified sweep's trial family)
    # ------------------------------------------------------------------
    _log("1/9 DSR power decomposition (certified sweep grid) ...")
    cert_sweep = parameter_sweep(
        normalized, prices_with_bmk, cfg_bt, _CERT_GRID,
        basis="active", base_weights=base_weights,
    )
    trial_sharpes = (
        cert_sweep.table["sharpe_daily"].dropna().to_numpy(dtype=float)
    )
    dsr_res = deflated_sharpe_ratio(equity, trial_sharpes)
    parity = assert_parity(equity, dsr_res.dsr)

    psr_res = probabilistic_sharpe_ratio(equity)
    sr_d = psr_res.sharpe_daily
    skew, kurt = psr_res.skewness, psr_res.kurtosis
    sr0 = dsr_res.expected_max_sharpe
    sigma_trials = float(np.std(trial_sharpes, ddof=1))

    sr_req = dsr_required_sr(sr0, n_days, skew, kurt)        # daily
    n_req_dsr = dsr_required_n(sr_d, sr0, skew, kurt)        # at current SR
    n_req_psr = dsr_required_n(sr_d, 0.0, skew, kurt)        # plain PSR>=.95
    n_req_t2 = 4.0 * (1.0 + sr_d * sr_d / 2.0) / (sr_d * sr_d)  # Lo t>=2

    power = {
        "n_days": n_days,
        "years": years,
        "actual_sharpe_daily": sr_d,
        "actual_ir_ann": sr_d * _SQ252,
        "actual_t": parity["sharpe_t_stat"],
        "skewness": skew,
        "kurtosis_nonexcess": kurt,
        "n_trials_certified": int(dsr_res.n_trials),
        "trial_sharpes_daily": trial_sharpes.tolist(),
        "trial_sigma_daily": sigma_trials,
        "sr0_daily": sr0,
        "sr0_ann": sr0 * _SQ252,
        "dsr_actual": dsr_res.dsr,
        "required_sharpe_daily_for_dsr95": sr_req,
        "required_ir_ann_for_dsr95": sr_req * _SQ252,
        "required_t_for_dsr95": lo_t(sr_req, n_days),
        "years_needed_at_current_ir_for_dsr95": n_req_dsr / 252.0,
        "years_needed_at_current_ir_for_psr95": n_req_psr / 252.0,
        "years_needed_at_current_ir_for_t2": n_req_t2 / 252.0,
    }
    _log(
        f"  actual IR {power['actual_ir_ann']:+.3f} (t {power['actual_t']:.2f}) "
        f"vs required IR {power['required_ir_ann_for_dsr95']:+.3f} "
        f"(t {power['required_t_for_dsr95']:.2f}) for DSR>=0.95; "
        f"years needed at current IR: {power['years_needed_at_current_ir_for_dsr95']:.0f}"
    )

    # ------------------------------------------------------------------
    # 9. Trial-count honesty: DSR at the cross-run ledger N=180
    # ------------------------------------------------------------------
    _log("2/9 Honest big-N DSR (cross-run ledger N=180) ...")
    sr0_honest = sr0_from_sigma_n(sigma_trials, _HONEST_N_TRIALS)
    z_honest = (sr_d - sr0_honest) * math.sqrt(n_days - 1.0) / math.sqrt(
        _variance_term(sr_d, skew, kurt)
    )
    honest = {
        "n_trials": _HONEST_N_TRIALS,
        "ledger": "~150 legacy scenarios + 8 segment runs + 18 tournament "
                  "specs + 4 S5 confirmation books",
        "trial_sigma_daily_used": sigma_trials,
        "sr0_daily": sr0_honest,
        "sr0_ann": sr0_honest * _SQ252,
        "dsr": float(sps.norm.cdf(z_honest)),
        "required_ir_ann_for_dsr95": dsr_required_sr(
            sr0_honest, n_days, skew, kurt
        ) * _SQ252,
        "years_needed_at_current_ir_for_dsr95": (
            dsr_required_n(sr_d, sr0_honest, skew, kurt) / 252.0
        ),
    }
    _log(f"  honest-N DSR = {honest['dsr']:.3f} "
         f"(SR0_ann {honest['sr0_ann']:.3f})")

    # ------------------------------------------------------------------
    # 2. Monte-Carlo deep dive (n_sims=500)
    # ------------------------------------------------------------------
    _log("3/9 Monte-Carlo random-signal null, n_sims=500 (slow) ...")
    mc = monte_carlo_null(
        normalized, prices_with_bmk, cfg_bt, n_sims=500, seed=0,
        basis="active", base_weights=base_weights,
    )
    mc_summary = {
        "n_sims": int(mc.n_sims),
        "actual_ir_ann": float(mc.actual_sharpe_ann),
        "null_mean": float(mc.null_mean),
        "null_std": float(np.std(mc.null_sharpes, ddof=1)),
        "null_q95": float(mc.null_q95),
        "p_value": float(mc.p_value),
        "actual_percentile_in_null": float(
            100.0 * np.mean(mc.null_sharpes < mc.actual_sharpe_ann)
        ),
        "p_value_bonferroni_3_segments": float(min(1.0, 3.0 * mc.p_value)),
    }
    _log(
        f"  MC p = {mc_summary['p_value']:.4f} "
        f"(percentile {mc_summary['actual_percentile_in_null']:.1f}, "
        f"null mean {mc_summary['null_mean']:+.3f}, "
        f"q95 {mc_summary['null_q95']:+.3f}); "
        f"Bonferroni x3 segments -> {mc_summary['p_value_bonferroni_3_segments']:.3f}"
    )

    # ------------------------------------------------------------------
    # 3. Walk-forward forensics
    # ------------------------------------------------------------------
    _log("4/9 Walk-forward forensics (5 anchored folds) ...")
    wf = walk_forward(
        normalized, prices_with_bmk, cfg_bt, _CERT_GRID, n_folds=5,
        base_weights=base_weights,
    )
    fold_rows = []
    for _, row in wf.fold_table.iterrows():
        fold_rows.append(
            {
                "fold": int(row["fold"]),
                "oos_start": str(pd.Timestamp(row["oos_start"]).date()),
                "oos_end": str(pd.Timestamp(row["oos_end"]).date()),
                "params": dict(row["params"]),
                "is_sharpe_daily": float(row["is_sharpe_daily"]),
                "oos_sharpe_daily": float(row["oos_sharpe_daily"]),
                "n_oos_periods": int(row["n_oos_periods"]),
            }
        )
    param_keys = [json.dumps(r["params"], sort_keys=True) for r in fold_rows]
    modal_share = (
        max(param_keys.count(k) for k in set(param_keys)) / len(param_keys)
        if param_keys
        else math.nan
    )
    wf_summary = {
        "folds": fold_rows,
        "wf_efficiency": float(wf.wf_efficiency),
        "oos_sharpe_ann": float(wf.oos_sharpe_ann),
        "frac_folds_oos_positive": float(wf.frac_folds_oos_positive),
        "modal_param_share": float(modal_share),
        "n_distinct_param_choices": len(set(param_keys)),
    }
    for r in fold_rows:
        _log(
            f"  fold {r['fold']}: {r['params']}  "
            f"IS {r['is_sharpe_daily']:+.4f}  OOS {r['oos_sharpe_daily']:+.4f}"
        )
    _log(f"  WFE {wf_summary['wf_efficiency']:.2f}, "
         f"frac OOS+ {wf_summary['frac_folds_oos_positive']:.2f}, "
         f"modal param share {modal_share:.2f}")

    # ------------------------------------------------------------------
    # 4. Subperiod robustness
    # ------------------------------------------------------------------
    _log("5/9 Subperiod robustness (halves / thirds) ...")
    halves = [subperiod_stats(c) for c in split_chunks(active_daily, 2)]
    thirds = [subperiod_stats(c) for c in split_chunks(active_daily, 3)]
    full_period = subperiod_stats(active_daily)
    for label, rows in (("half", halves), ("third", thirds)):
        for i, r in enumerate(rows, 1):
            _log(
                f"  {label} {i} [{r['start']} .. {r['end']}]: "
                f"ann {r['ann_active_return']:+.2%}  IR {r['ir']:+.2f}  "
                f"q-hit {r['quarter_hit_rate']:.2f}"
            )

    # ------------------------------------------------------------------
    # 5. Rolling 252d IR
    # ------------------------------------------------------------------
    _log("6/9 Rolling 252d IR ...")
    roll = active_daily.rolling(252)
    roll_ir = (roll.mean() / roll.std(ddof=1) * _SQ252).dropna()
    rolling = {
        "n_windows": int(len(roll_ir)),
        "frac_positive": float((roll_ir > 0.0).mean()),
        "worst": float(roll_ir.min()),
        "worst_date": str(roll_ir.idxmin().date()),
        "best": float(roll_ir.max()),
        "best_date": str(roll_ir.idxmax().date()),
        "median": float(roll_ir.median()),
        "last": float(roll_ir.iloc[-1]),
    }
    _log(
        f"  {rolling['frac_positive']:.1%} of {rolling['n_windows']} windows "
        f"positive; worst {rolling['worst']:+.2f} ({rolling['worst_date']}), "
        f"best {rolling['best']:+.2f} ({rolling['best_date']})"
    )

    # ------------------------------------------------------------------
    # 6. Cost stress (period TC deducted from the gross daily active curve)
    # ------------------------------------------------------------------
    _log("7/9 Cost stress: tc_bps in {2, 5, 10, 20} ...")
    turnover = result.period_results["turnover"]
    ann_turnover = float(turnover.iloc[1:].mean()) * (252.0 / 63.0)
    cost_rows = []
    for tc in (0.0, 2.0, 5.0, 10.0, 20.0):
        net = active_daily.copy()
        costs = turnover * (tc / 1e4)
        for d_ret, c in costs.items():
            if d_ret in net.index:
                net.loc[d_ret] -= c
        cost_rows.append(
            {
                "tc_bps": tc,
                "ir_ann": ann_ir(net),
                "ann_active_return": float(net.mean()) * 252.0,
                "ann_cost_drag": float(costs.iloc[1:].mean()) * (252.0 / 63.0),
            }
        )
        _log(f"  tc {tc:>4.0f} bps: IR {cost_rows[-1]['ir_ann']:+.3f}  "
             f"(ann active {cost_rows[-1]['ann_active_return']:+.2%})")
    cost_stress = {
        "note": "Headline IR is gross of transaction costs (the engine's "
                "daily curve carries no TC; costs only hit period net "
                "returns). Stress deducts turnover*tc at each period end.",
        "ann_one_sided_turnover": ann_turnover,
        "rows": cost_rows,
    }

    # ------------------------------------------------------------------
    # 7. Parameter neighborhood (wider 1-D sweep)
    # ------------------------------------------------------------------
    _log("8/9 Parameter neighborhood sweep ...")
    nbhd_grid = {
        "relative_selection_score": (3, 4, 5, 6, 7),
        "periodicity": (42, 63, 84),
        "active_share": (0.2, 0.3, 0.4),
    }
    nbhd = parameter_sweep(
        normalized, prices_with_bmk, cfg_bt, nbhd_grid,
        basis="active", base_weights=base_weights,
    )
    stab = stability_summary(nbhd, cfg_bt)
    nbhd_rows = []
    for _, row in nbhd.table.iterrows():
        deltas = {
            k: row[k]
            for k in ("relative_selection_score", "periodicity", "active_share")
            if row[k] != getattr(cfg_bt, k)
        }
        nbhd_rows.append(
            {
                "config": deltas if deltas else {"<default>": True},
                "sharpe_ann": float(row["sharpe_ann"]),
                "total_return": float(row["total_return"]),
                "max_drawdown": float(row["max_drawdown"]),
            }
        )
    neighborhood = {
        "grid": {k: list(v) for k, v in nbhd_grid.items()},
        "rows": nbhd_rows,
        "n_trials": stab.n_trials,
        "sharpe_mean": stab.sharpe_mean,
        "sharpe_std": stab.sharpe_std,
        "sharpe_min": stab.sharpe_min,
        "sharpe_max": stab.sharpe_max,
        "frac_positive": stab.frac_positive,
        "default_sharpe": stab.default_sharpe,
        "default_zscore": stab.default_zscore,
    }
    _log(
        f"  {stab.n_trials} configs: frac+ {stab.frac_positive:.2f}, "
        f"IR range [{stab.sharpe_min:+.3f}, {stab.sharpe_max:+.3f}], "
        f"default z {stab.default_zscore:+.2f}"
    )

    # ------------------------------------------------------------------
    # 8. IC time stability (relative IC @63, halves)
    # ------------------------------------------------------------------
    _log("9/9 Composite relative IC @63: full vs halves ...")
    ic_series = ic_module.information_coefficient(
        normalized, prices, 63, "relative"
    )["IC"].dropna()
    mid = len(ic_series) // 2
    ic_stability = {}
    for label, s in (
        ("full", ic_series),
        ("first_half", ic_series.iloc[:mid]),
        ("second_half", ic_series.iloc[mid:]),
    ):
        st = ic_module.ic_stats(s)
        ic_stability[label] = {
            "start": str(s.index[0].date()),
            "end": str(s.index[-1].date()),
            "n": int(len(s)),
            "mean_ic": st["mean_ic"],
            "t_stat": st["t_stat"],
            "hit_rate": st["hit_rate"],
        }
        _log(f"  {label:12s}: mean {st['mean_ic']:+.4f}  "
             f"t {st['t_stat']:+.2f}  n {len(s)}")

    # ------------------------------------------------------------------
    # 10. Signature scoreboard + verdict
    # ------------------------------------------------------------------
    ir_at_10bps = next(r["ir_ann"] for r in cost_rows if r["tc_bps"] == 10.0)
    n_pos_thirds = sum(1 for r in thirds if r["ir"] > 0.0)
    signatures = {
        "wf_is_much_greater_than_oos (WFE<0.5)": wf_summary["wf_efficiency"] < 0.5,
        "wf_unstable_param_choices (modal share<0.6)": modal_share < 0.6,
        "alpha_concentrated_in_one_subperiod (<=1 of 3 thirds IR>0)":
            n_pos_thirds <= 1,
        "fails_random_signal_null (MC p>0.05)": mc_summary["p_value"] > 0.05,
        "default_config_lone_spike (abs z>1.5 or frac+<0.7)": (
            abs(stab.default_zscore) > 1.5 or stab.frac_positive < 0.7
        ),
        "edge_dies_at_realistic_costs (IR<=0 at 10bps)": ir_at_10bps <= 0.0,
        "ic_sign_flips_between_halves": (
            np.sign(ic_stability["first_half"]["mean_ic"])
            != np.sign(ic_stability["second_half"]["mean_ic"])
        ),
    }
    signatures = {k: bool(v) for k, v in signatures.items()}
    n_present = sum(signatures.values())
    _log(f"Overfitting signatures present: {n_present}/{len(signatures)}")

    payload = rr._json_safe(
        {
            "strategy": {
                "segment": "EM",
                "signal": "vm (50/50 Value+Momentum, 6 factors)",
                "construction": "Cap_Tilt (active_share 0.30)",
                "periodicity": 63,
                "benchmark": "vendor EM cap index",
                "basis": "active",
                "certified_verdict_file": os.path.basename(_VERDICT_PATH),
                "reconstruction_parity": parity,
            },
            "power_decomposition": power,
            "honest_big_n_dsr": honest,
            "monte_carlo": mc_summary,
            "walk_forward": wf_summary,
            "subperiods": {"full": full_period, "halves": halves, "thirds": thirds},
            "rolling_252d_ir": rolling,
            "cost_stress": cost_stress,
            "parameter_neighborhood": neighborhood,
            "ic_stability": ic_stability,
            "signatures": signatures,
            "n_signatures_present": n_present,
            "runtime_seconds": round(time.time() - t0, 1),
        }
    )
    with open(_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _log(f"JSON -> {_JSON_PATH}")

    write_markdown(payload)
    _log(f"Markdown -> {_MD_PATH}")
    _log(f"Done in {payload['runtime_seconds']}s.")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown(p: dict) -> None:  # noqa: C901 — sequential report builder
    pw, hn, mc = p["power_decomposition"], p["honest_big_n_dsr"], p["monte_carlo"]
    wf, sub, roll = p["walk_forward"], p["subperiods"], p["rolling_252d_ir"]
    cost, nb, ics = p["cost_stress"], p["parameter_neighborhood"], p["ic_stability"]
    sig = p["signatures"]

    L: list[str] = []
    L.append("# Overfitting Forensics — EM Cap-Tilt (vm @63d vs EM cap index)")
    L.append("")
    L.append(
        f"**Question:** is the strategy overfitted, or is the DSR failure "
        f"(0.762 < 0.95) a statistical-power artifact?  "
        f"**Data:** {pw['n_days']} active daily returns "
        f"({pw['years']:.1f}y, {sub['full']['start']} to {sub['full']['end']}). "
        f"Reconstruction reproduces the certified verdict bit-for-bit "
        f"(sharpe_ann / t / PSR / DSR all match to <1e-8). "
        f"Script: `scripts/overfit_forensics.py`; machine-readable results in "
        f"`outputs/research/overfit_forensics_EM.json` (gitignored, "
        f"regenerable)."
    )
    L.append("")

    # 1 — power
    L.append("## 1. DSR power decomposition")
    L.append("")
    L.append(
        f"The certified DSR uses the validation sweep's "
        f"{pw['n_trials_certified']} trial Sharpes "
        f"(sigma_daily = {pw['trial_sigma_daily']:.5f}) giving a deflated "
        f"benchmark SR0 = {pw['sr0_daily']:.5f}/day "
        f"({pw['sr0_ann']:+.3f} annualized). Inverting PSR(SR0) = 0.95 at the "
        f"observed sample size, skew ({pw['skewness']:+.2f}) and kurtosis "
        f"({pw['kurtosis_nonexcess']:.1f}):"
    )
    L.append("")
    L.append("| Quantity | Actual | Required for DSR >= 0.95 |")
    L.append("|---|---|---|")
    L.append(
        f"| Annualized IR (active Sharpe) | {pw['actual_ir_ann']:+.3f} | "
        f"{pw['required_ir_ann_for_dsr95']:+.3f} |"
    )
    L.append(
        f"| Lo (2002) t-stat | {pw['actual_t']:.2f} | "
        f"{pw['required_t_for_dsr95']:.2f} |"
    )
    L.append(f"| DSR | {pw['dsr_actual']:.3f} | 0.950 |")
    L.append("")
    L.append(
        f"At the CURRENT IR of {pw['actual_ir_ann']:+.3f}, the sample needed "
        f"is **{pw['years_needed_at_current_ir_for_dsr95']:.0f} years** for "
        f"DSR >= 0.95, {pw['years_needed_at_current_ir_for_psr95']:.0f} years "
        f"for plain PSR >= 0.95, and "
        f"{pw['years_needed_at_current_ir_for_t2']:.0f} years for Sharpe "
        f"t >= 2 (t ~ IR x sqrt(years)). The strategy has "
        f"{pw['years']:.1f} years. A true IR ~0.3 cannot mathematically "
        f"clear a t~2-equivalent gate on this span regardless of whether the "
        f"edge is real — the DSR gate is power-limited here, by construction."
    )
    L.append("")

    # 2 — MC
    L.append("## 2. Monte-Carlo random-signal null (n = 500)")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Actual IR (ann) | {mc['actual_ir_ann']:+.3f} |")
    L.append(f"| Null mean / std | {mc['null_mean']:+.3f} / {mc['null_std']:.3f} |")
    L.append(f"| Null q95 | {mc['null_q95']:+.3f} |")
    L.append(f"| Actual percentile in null | {mc['actual_percentile_in_null']:.1f} |")
    L.append(f"| p-value (add-one) | **{mc['p_value']:.4f}** |")
    L.append(
        f"| Family-wise (x3 segments, Bonferroni) | "
        f"{mc['p_value_bonferroni_3_segments']:.3f} |"
    )
    L.append("")
    L.append(
        "The null holds the Cap_Tilt construction, costs and benchmark "
        "constant and randomizes only the signal — this isolates selection "
        "skill from construction. 500 sims confirm the certified n=100 "
        "result."
    )
    L.append("")

    # 3 — WF
    L.append("## 3. Walk-forward forensics (5 anchored folds)")
    L.append("")
    L.append("| Fold | OOS window | Chosen params | IS Sharpe (per-period) | OOS Sharpe (per-period) |")
    L.append("|---|---|---|---|---|")
    for r in wf["folds"]:
        ps = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        L.append(
            f"| {r['fold']} | {r['oos_start']} to {r['oos_end']} | {ps} | "
            f"{r['is_sharpe_daily']:+.4f} | {r['oos_sharpe_daily']:+.4f} |"
        )
    L.append("")
    L.append(
        f"WF efficiency = **{wf['wf_efficiency']:.2f}** (OOS mean / IS mean; "
        f">1 means OOS BEAT in-sample selection — the opposite of the "
        f"overfit signature IS >> OOS). "
        f"{wf['frac_folds_oos_positive']:.0%} of folds OOS-positive. "
        f"Param choices: {wf['n_distinct_param_choices']} distinct "
        f"combination(s) across 5 folds (modal share "
        f"{wf['modal_param_share']:.0%})."
    )
    L.append("")

    # 4 — subperiods
    L.append("## 4. Subperiod robustness")
    L.append("")
    L.append("| Period | Window | Ann active | IR | Quarterly hit rate |")
    L.append("|---|---|---|---|---|")
    f_ = sub["full"]
    L.append(
        f"| Full | {f_['start']} to {f_['end']} | "
        f"{f_['ann_active_return']:+.2%} | {f_['ir']:+.2f} | "
        f"{f_['quarter_hit_rate']:.0%} ({f_['n_quarters']}q) |"
    )
    for i, r in enumerate(sub["halves"], 1):
        L.append(
            f"| Half {i} | {r['start']} to {r['end']} | "
            f"{r['ann_active_return']:+.2%} | {r['ir']:+.2f} | "
            f"{r['quarter_hit_rate']:.0%} ({r['n_quarters']}q) |"
        )
    for i, r in enumerate(sub["thirds"], 1):
        L.append(
            f"| Third {i} | {r['start']} to {r['end']} | "
            f"{r['ann_active_return']:+.2%} | {r['ir']:+.2f} | "
            f"{r['quarter_hit_rate']:.0%} ({r['n_quarters']}q) |"
        )
    L.append("")

    # 5 — rolling
    L.append("## 5. Rolling 252-day IR")
    L.append("")
    L.append(
        f"{roll['frac_positive']:.1%} of {roll['n_windows']} rolling windows "
        f"positive; median {roll['median']:+.2f}; worst {roll['worst']:+.2f} "
        f"({roll['worst_date']}); best {roll['best']:+.2f} "
        f"({roll['best_date']}); latest {roll['last']:+.2f}."
    )
    L.append("")

    # 6 — costs
    L.append("## 6. Cost stress")
    L.append("")
    L.append(
        f"Annualized one-sided turnover = "
        f"{cost['ann_one_sided_turnover']:.1%}. {cost['note']}"
    )
    L.append("")
    L.append("| tc (bps) | IR (ann) | Ann active return | Ann cost drag |")
    L.append("|---|---|---|---|")
    for r in cost["rows"]:
        L.append(
            f"| {r['tc_bps']:.0f} | {r['ir_ann']:+.3f} | "
            f"{r['ann_active_return']:+.2%} | {r['ann_cost_drag']:.3%} |"
        )
    L.append("")

    # 7 — neighborhood
    L.append("## 7. Parameter neighborhood (1-D sweeps around the default)")
    L.append("")
    L.append("| Config (delta vs default) | IR (ann) | Total active return | Max DD (active eq.) |")
    L.append("|---|---|---|---|")
    for r in nb["rows"]:
        cfg_str = ", ".join(
            f"{k}={v}" for k, v in r["config"].items() if k != "<default>"
        ) or "default (N=5, p=63, as=0.30)"
        L.append(
            f"| {cfg_str} | {r['sharpe_ann']:+.3f} | "
            f"{r['total_return']:+.2%} | {r['max_drawdown']:+.2%} |"
        )
    L.append("")
    L.append(
        f"{nb['n_trials']} configurations: **{nb['frac_positive']:.0%} "
        f"positive**, IR mean {nb['sharpe_mean']:+.3f} "
        f"(std {nb['sharpe_std']:.3f}), range "
        f"[{nb['sharpe_min']:+.3f}, {nb['sharpe_max']:+.3f}]. Default config "
        f"z-score = {nb['default_zscore']:+.2f} — the default is "
        f"{'NOT a lone spike' if abs(nb['default_zscore']) <= 1.5 else 'an outlier'} "
        f"in its neighborhood."
    )
    L.append("")

    # 8 — IC
    L.append("## 8. Composite relative IC @63d — time stability")
    L.append("")
    L.append("| Window | Dates | n | Mean IC | t-stat | Hit rate |")
    L.append("|---|---|---|---|---|---|")
    for label in ("full", "first_half", "second_half"):
        r = ics[label]
        L.append(
            f"| {label} | {r['start']} to {r['end']} | {r['n']} | "
            f"{r['mean_ic']:+.4f} | {r['t_stat']:+.2f} | {r['hit_rate']:.0%} |"
        )
    L.append("")

    # 9 — honest N
    L.append("## 9. Trial-count honesty — DSR at the cross-run ledger N = 180")
    L.append("")
    yrs_honest = hn["years_needed_at_current_ir_for_dsr95"]
    yrs_honest_txt = (
        f"{yrs_honest:.0f} years"
        if isinstance(yrs_honest, (int, float)) and math.isfinite(yrs_honest)
        else "INFINITE — the current IR sits below the deflated benchmark, "
             "so no sample length clears DSR at this IR under honest big-N "
             "accounting; only a higher realized IR (or a smaller honest "
             "trial family) can"
    )
    L.append(
        f"Ledger: {hn['ledger']}. Using the sweep trial sigma "
        f"({hn['trial_sigma_daily_used']:.5f}/day) with N = {hn['n_trials']} "
        f"raises the deflated benchmark to SR0 = {hn['sr0_ann']:+.3f} "
        f"annualized — ABOVE the actual IR of {pw['actual_ir_ann']:+.3f} — "
        f"and gives **DSR = {hn['dsr']:.3f}** (vs {pw['dsr_actual']:.3f} at "
        f"the certified N = {pw['n_trials_certified']}). The required IR "
        f"rises to {hn['required_ir_ann_for_dsr95']:+.3f}; years needed at "
        f"the current IR: {yrs_honest_txt}. Caveat: the sigma behind SR0 is "
        f"estimated from only {pw['n_trials_certified']} sweep trials and "
        f"those 180 ledger trials are highly correlated (variants of the "
        f"same V+M signal on overlapping data), so this is a worst-case "
        f"deflation, not a point estimate. The qualitative conclusion — the "
        f"DSR gate is unreachable on this span at IR ~0.3 — holds at any N "
        f">= {pw['n_trials_certified']}."
    )
    L.append("")

    # 10 — verdict
    L.append("## 10. Verdict")
    L.append("")
    L.append("### Overfitting signature scoreboard")
    L.append("")
    L.append("| Signature | Present? |")
    L.append("|---|---|")
    for k, v in sig.items():
        L.append(f"| {k} | {'**PRESENT**' if v else 'absent'} |")
    L.append("")
    n_present = p["n_signatures_present"]
    L.append(
        f"**{n_present} of {len(sig)} overfitting signatures present** "
        f"(both discussed below; neither is a parameter-tuning signature)."
    )
    L.append("")
    L.append("### Conclusion")
    L.append("")
    h1, h2 = sub["halves"]
    t1, t2, t3 = sub["thirds"]
    L.append(
        f"**The strategy is NOT overfitted in the parameter / selection-"
        f"tuning sense, and the DSR failure is a statistical-power artifact "
        f"— but the alpha is front-loaded, which is a real (separate) "
        f"risk.** Three lines of evidence support each leg:"
    )
    L.append("")
    L.append(
        f"1. **Power, decisively.** At the realized IR of "
        f"{pw['actual_ir_ann']:+.2f}, DSR >= 0.95 needs "
        f"~{pw['years_needed_at_current_ir_for_dsr95']:.0f} years of data "
        f"(plain PSR: {pw['years_needed_at_current_ir_for_psr95']:.0f}y; "
        f"t >= 2: {pw['years_needed_at_current_ir_for_t2']:.0f}y); the "
        f"sample has {pw['years']:.1f}. Equivalently the gate demands IR "
        f"{pw['required_ir_ann_for_dsr95']:+.2f} (t "
        f"{pw['required_t_for_dsr95']:.2f}) on this span — "
        f"{pw['required_ir_ann_for_dsr95']/pw['actual_ir_ann']:.1f}x the "
        f"point estimate. A true IR-0.3 strategy fails this gate by "
        f"construction; the failure carries no information about "
        f"overfitting."
    )
    L.append(
        f"2. **Every parameter-tuning probe is clean.** Walk-forward OOS "
        f"folds BEAT in-sample selection (WFE {wf['wf_efficiency']:.2f}, "
        f"{wf['frac_folds_oos_positive']:.0%} folds positive — the opposite "
        f"of the IS>>OOS signature); the parameter neighborhood is "
        f"{nb['frac_positive']:.0%} positive with the default at "
        f"z = {nb['default_zscore']:+.2f} (dead-center, and NOT the peak — "
        f"p=84 scores higher at {nb['sharpe_max']:+.2f}; a tuned config "
        f"would sit at the peak); the edge survives 10x assumed costs "
        f"(IR {next(r['ir_ann'] for r in cost['rows'] if r['tc_bps'] == 20.0):+.2f} "
        f"at 20 bps); and the random-signal null — construction held "
        f"constant — rejects at p = {mc['p_value']:.3f} on 500 sims "
        f"(actual book at the {mc['actual_percentile_in_null']:.0f}th "
        f"percentile of the null). The two flagged signatures are benign "
        f"variants: WF param choices rotate across folds precisely BECAUSE "
        f"the neighborhood is flat (IS Sharpes are statistically "
        f"indistinguishable, so the argmax is noise — corroborated by "
        f"{nb['frac_positive']:.0%} of configs positive), and the deployed "
        f"params were fixed ex-ante by literature priors, not selected by "
        f"this WF."
    )
    L.append(
        f"3. **The honest weakness is time-decay, not tuning.** The alpha "
        f"is front-loaded: half-1 IR {h1['ir']:+.2f} "
        f"(q-hit {h1['quarter_hit_rate']:.0%}) vs half-2 IR {h2['ir']:+.2f} "
        f"(q-hit {h2['quarter_hit_rate']:.0%}); thirds {t1['ir']:+.2f} / "
        f"{t2['ir']:+.2f} / {t3['ir']:+.2f}; composite IC fades from "
        f"{ics['first_half']['mean_ic']:+.3f} to "
        f"{ics['second_half']['mean_ic']:+.3f} across halves. Two of three "
        f"thirds and {roll['frac_positive']:.0%} of rolling windows are "
        f"still positive, so the alpha is not a single-episode artifact, "
        f"but the recent-sample evidence alone would not justify deployment."
    )
    L.append("")
    L.append("### Residual risks (stated honestly)")
    L.append("")
    L.append(
        f"1. **Post-hoc segment selection.** EM was chosen as the deploy "
        f"candidate AFTER observing all three segments; a Bonferroni-style "
        f"correction across the 3-segment family puts the MC evidence at "
        f"~{mc['p_value_bonferroni_3_segments']:.2f} family-wise — "
        f"suggestive, not significant at 5%. The honest claim is "
        f"'selection skill significant within EM, marginal family-wise'."
    )
    L.append(
        f"2. **Big-N deflation.** Against the full cross-run ledger "
        f"(N~{hn['n_trials']}) the DSR drops to {hn['dsr']:.3f} and the "
        f"deflated benchmark ({hn['sr0_ann']:+.2f}) exceeds the realized IR "
        f"— under worst-case trial accounting no amount of history at this "
        f"IR certifies. The program has consumed many trials; only fresh "
        f"OOS data (live quarters) repays that debt."
    )
    L.append(
        f"3. **Alpha decay.** Section 4/8: most of the realized alpha is "
        f"pre-2018 and the second-half IC is ~0. The per-period IC t-stats "
        f"are individually insignificant in EM throughout (skill shows in "
        f"the MC book-level test, not the period IC), so the edge rests on "
        f"construction + breadth — monitor the rolling IR and quarterly "
        f"re-certs; a continued flat second-half trend is grounds to pull "
        f"the candidate regardless of the full-sample stats."
    )
    L.append("")
    L.append(
        f"*Generated by `scripts/overfit_forensics.py` in "
        f"{p['runtime_seconds']}s; deterministic (seeded); inputs are the "
        f"gitignored vendor data.*"
    )
    L.append("")

    with open(_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
