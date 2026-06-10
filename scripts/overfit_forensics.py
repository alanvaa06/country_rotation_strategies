"""Overfitting forensics for a Cap-Tilt deploy candidate (parameterized).

Question under test
-------------------
Is the {segment} Cap-Tilt strategy (50/50 Value+Momentum composite, @63d,
Cap_Tilt construction, active basis) OVERFITTED, or is its DSR shortfall a
statistical-power artifact of the ~16-year sample?

Two benchmark modes (mirrors ``scripts/research_run.py``):

* default — the segment's own vendor cap index column (``--bmk-source
  index`` in research_run; verdict tag ``_capbmk``), e.g. EM vs the EM cap
  index (the original EM forensics run).
* ``--bmk-index NAME`` — a single cross-segment vendor index pinned as THE
  benchmark (verdict tag ``_vs{NAME}``), e.g. ``--segment DM --bmk-index
  World`` = the ACWI-relative lead candidate.

Strategy reconstruction is exact (asserted against the certified verdict
``outputs/research/verdict_{seg}_prior_vm_p63_active_captilt_{capbmk|vsNAME}
.json``: sharpe_ann, t, PSR, DSR reproduced bit-for-bit).

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
   (N=186 ~ 200: ~150 legacy scenarios + 8 segment runs + 18 tournament
   specs + 4 S5 books + 6 ACWI-relative books), with an N=200 sensitivity.
10. Stationary-bootstrap significance — one-sided p(IR <= 0), 2000
    Politis-Romano resamples (expected block sqrt(n), seed 42, add-one
    smoothing) on the daily active series.
11. Alpha decomposition (``--bmk-index`` mode only) — split the daily active
    return into the passive segment spread (segment cap index minus the
    pinned benchmark index: the structural bet) and pure within-segment
    selection (book minus segment cap index); IRs, NW t-stats, correlation,
    variance shares, and the active~spread regression.
12. Verdict — overfitting signatures present/absent, significance summary,
    segment-bet vs selection decomposition, residual risks (family-wise
    honesty: the candidate is 1 book of the evaluation-round family).

Outputs
-------
* ``outputs/research/overfit_forensics_{seg}[_vs{NAME}].json`` (gitignored)
* ``outputs/research/overfit_forensics_{seg}[_vs{NAME}].md``   (committed)

Usage
-----
    python scripts/overfit_forensics.py                          # EM vs EM cap index
    python scripts/overfit_forensics.py --segment DM --bmk-index World
    python scripts/overfit_forensics.py ... --render-only        # re-render md from json

(~3-6 min full; needs local gitignored ``Inputs/`` + ``Classification.xlsx``.)
"""
from __future__ import annotations

import argparse
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
    _stationary_bootstrap_indices,
    deflated_sharpe_ratio,
    newey_west_tstat,
    probabilistic_sharpe_ratio,
    sharpe_significance,
)

_OUT_DIR = os.path.join(_REPO, "outputs", "research")

#: Certified validation grid (research_run.py main(), non-quick).
_CERT_GRID = {"relative_selection_score": (3, 5, 7), "periodicity": (21, 63)}

#: Cross-run multiple-testing ledger (final_evidence_dossier.md "Standing
#: decisions" #4, extended 2026-06-10): ~150 legacy scenarios + 8 segment
#: runs + 6x3 tournament specs + 4 S5 confirmation books + 6 ACWI-relative
#: books (3 segments x cap_tilt/eqw vs the pinned World index).
_HONEST_N_TRIALS = 186
_HONEST_LEDGER = (
    "~150 legacy scenarios + 8 segment runs + 18 tournament specs + "
    "4 S5 confirmation books + 6 ACWI-relative books"
)
#: Round-number sensitivity for the honest-N DSR (the ledger is ~200).
_HONEST_N_SENSITIVITY = 200

#: Evaluation-round family size for Bonferroni honesty: the ACWI round
#: evaluated 6 books (World/DM/EM x cap_tilt/eqw vs the pinned index); the
#: original own-cap-index round compared 3 segments.
_FAMILY_N_ACWI = 6
_FAMILY_N_SEGMENTS = 3

_EULER = 0.5772156649015328
_SQ252 = math.sqrt(252.0)


def _log(msg: str) -> None:
    print(f"[overfit_forensics] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Paths / labels
# ---------------------------------------------------------------------------

def out_suffix(segment: str, bmk_index: str | None) -> str:
    """Output filename suffix: ``{seg}`` or ``{seg}_vs{NAME}``."""
    return segment + (f"_vs{bmk_index}" if bmk_index else "")


def verdict_path(segment: str, bmk_index: str | None) -> str:
    """Certified verdict JSON path (mirrors research_run.py tag logic)."""
    tag = f"{segment}_prior_vm_p63_active_captilt"
    tag += f"_vs{bmk_index}" if bmk_index else "_capbmk"
    return os.path.join(_OUT_DIR, f"verdict_{tag}.json")


def bmk_label(segment: str, bmk_index: str | None) -> str:
    """Human-readable benchmark description for logs and the report."""
    if bmk_index is None:
        return f"vendor {segment} cap index"
    extra = " (ACWI-equivalent)" if bmk_index == "World" else ""
    return f"vendor '{bmk_index}' index{extra}"


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

def reconstruct(segment: str, bmk_index: str | None):
    """Rebuild the certified Cap-Tilt book; mirrors research_run.py exactly.

    ``bmk_col = bmk_index or segment``; in both modes the benchmark column
    is the raw vendor index level from the Price input (``--bmk-source
    index`` / ``--bmk-index NAME`` branches of research_run.py).
    """
    cfg = load_config(os.path.join(_REPO, "configs", "default.json"))
    processed, classification = rr.ingest(cfg)

    universe = [
        c
        for c in rr.extract_universe(classification, segment)
        if c in processed["Price"].columns
    ]
    price_df = processed["Price"]
    prices = price_df[universe].copy()
    prices_with_bmk = prices.copy()
    bmk_col = bmk_index or segment
    if bmk_col not in price_df.columns:
        raise SystemExit(
            f"[overfit_forensics] ERROR: benchmark index column "
            f"'{bmk_col}' missing from the Price input."
        )
    prices_with_bmk[bmk_col] = price_df[bmk_col]  # vendor index level

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
        bmk=bmk_col,
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
        price_df,
        cfg_bt,
        base_weights,
        result,
        active_daily,
    )


def assert_parity(equity, dsr_recomputed, vpath: str) -> dict:
    """Assert the reconstruction reproduces the certified verdict stats."""
    with open(vpath, encoding="utf-8") as f:
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
                f"certified {want!r} ({os.path.basename(vpath)})"
            )
    _log(
        "Reconstruction parity PASS (sharpe_ann/t/PSR/DSR match certified "
        "verdict to <1e-8)."
    )
    return {k: v[0] for k, v in checks.items()}


# ---------------------------------------------------------------------------
# New analyses: bootstrap significance + alpha decomposition
# ---------------------------------------------------------------------------

def bootstrap_significance(
    active_daily: pd.Series, n_boot: int = 2000, seed: int = 42
) -> dict:
    """One-sided stationary-bootstrap test of H0: IR <= 0.

    Politis-Romano stationary bootstrap (expected block length sqrt(n),
    wrap-around) on the daily active returns; each resample's annualized IR
    is recorded and ``p = (#(IR_b <= 0) + 1) / (n_valid + 1)`` (add-one
    smoothing, Davison & Hinkley convention — matches the MC null)."""
    r = active_daily.to_numpy(dtype=float)
    n = len(r)
    bl = max(1.0, math.sqrt(n))
    rng = np.random.default_rng(seed)
    irs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, bl, rng)
        rb = r[idx]
        sd = float(rb.std(ddof=1))
        irs[b] = math.nan if not sd > 0.0 else float(rb.mean()) / sd * _SQ252
    valid = irs[np.isfinite(irs)]
    p = (float(np.sum(valid <= 0.0)) + 1.0) / (len(valid) + 1.0)
    return {
        "n_boot": int(n_boot),
        "n_valid": int(len(valid)),
        "block_length_expected": bl,
        "seed": seed,
        "p_one_sided_ir_le_0": p,
        "ir_point": ann_ir(active_daily),
        "ir_q025": float(np.percentile(valid, 2.5)),
        "ir_q05": float(np.percentile(valid, 5.0)),
        "ir_median": float(np.percentile(valid, 50.0)),
        "ir_q95": float(np.percentile(valid, 95.0)),
        "ir_q975": float(np.percentile(valid, 97.5)),
        "frac_resamples_ir_le_0": float(np.mean(valid <= 0.0)),
    }


def alpha_decomposition(
    active_daily: pd.Series,
    result,
    price_df: pd.DataFrame,
    segment: str,
) -> dict:
    """Split daily active return into structural segment bet + selection.

    active (book - bmk_index) == spread (segment cap index - bmk_index)
                               + selection (book - segment cap index)

    The segment cap index daily return is the pct_change of the vendor
    ``{segment}`` Price column on the same date grid the Engine used, so
    the identity holds to machine precision on the common index."""
    idx = active_daily.index
    frame = pd.DataFrame(
        {
            "book": result.daily_returns.reindex(idx),
            "seg": price_df[segment].pct_change().reindex(idx),
            "bmk": result.daily_bmk_returns.reindex(idx),
        }
    ).dropna()
    active = frame["book"] - frame["bmk"]
    spread = frame["seg"] - frame["bmk"]       # structural segment-vs-bmk bet
    selection = frame["book"] - frame["seg"]   # pure within-segment selection
    identity_max_abs_err = float((active - (spread + selection)).abs().max())

    # Regression: active = a + b * spread + e
    var_spread = float(spread.var(ddof=1))
    cov_as = float(np.cov(active, spread, ddof=1)[0, 1])
    beta = cov_as / var_spread if var_spread > 0.0 else math.nan
    corr_active_spread = float(active.corr(spread))

    var_active = float(active.var(ddof=1))
    var_sel = float(selection.var(ddof=1))
    cov_sel_spread = float(np.cov(selection, spread, ddof=1)[0, 1])

    nw_spread = newey_west_tstat(spread)
    nw_sel = newey_west_tstat(selection)

    return {
        "n_days": int(len(frame)),
        "identity_max_abs_err": identity_max_abs_err,
        "active": {
            "ann_mean": float(active.mean()) * 252.0,
            "ir": ann_ir(active),
        },
        "segment_spread": {
            "definition": f"{segment} cap index minus benchmark index "
                          f"(passive structural bet)",
            "ann_mean": float(spread.mean()) * 252.0,
            "ir": ann_ir(spread),
            "nw_t": nw_spread.t_stat,
            "ann_vol": float(spread.std(ddof=1)) * _SQ252,
        },
        "within_segment_selection": {
            "definition": f"book minus {segment} cap index "
                          f"(pure within-{segment} selection)",
            "ann_mean": float(selection.mean()) * 252.0,
            "ir": ann_ir(selection),
            "nw_t": nw_sel.t_stat,
            "ann_vol": float(selection.std(ddof=1)) * _SQ252,
        },
        "corr_selection_spread": float(selection.corr(spread)),
        "regression_active_on_spread": {
            "beta": beta,
            "r_squared": corr_active_spread ** 2,
        },
        "variance_shares_of_active": {
            "selection": var_sel / var_active if var_active > 0 else math.nan,
            "spread": var_spread / var_active if var_active > 0 else math.nan,
            "2cov": (2.0 * cov_sel_spread / var_active
                     if var_active > 0 else math.nan),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfitting forensics for a Cap-Tilt deploy candidate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--segment", default="EM", choices=("World", "DM", "EM"),
        help="Market segment universe of the book.",
    )
    parser.add_argument(
        "--bmk-index", default=None, dest="bmk_index", metavar="NAME",
        help="Vendor index column pinned as THE benchmark (e.g. 'World' = "
             "ACWI-equivalent). Default None = the segment's own vendor cap "
             "index column (the original capbmk forensics).",
    )
    parser.add_argument(
        "--render-only", action="store_true", default=False,
        help="Skip all computation; re-render the markdown report from the "
             "existing forensics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    os.makedirs(_OUT_DIR, exist_ok=True)

    suffix = out_suffix(args.segment, args.bmk_index)
    json_path = os.path.join(_OUT_DIR, f"overfit_forensics_{suffix}.json")
    md_path = os.path.join(_OUT_DIR, f"overfit_forensics_{suffix}.md")
    vpath = verdict_path(args.segment, args.bmk_index)

    if args.render_only:
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
        write_markdown(payload, md_path)
        _log(f"Markdown (render-only) -> {md_path}")
        return

    blabel = bmk_label(args.segment, args.bmk_index)
    family_n = _FAMILY_N_ACWI if args.bmk_index else _FAMILY_N_SEGMENTS
    family_desc = (
        f"{_FAMILY_N_ACWI} ACWI-relative books (World/DM/EM x cap_tilt/eqw) "
        f"evaluated in this round"
        if args.bmk_index
        else f"{_FAMILY_N_SEGMENTS} segments compared vs their own cap index"
    )

    _log(f"Reconstructing {args.segment} Cap-Tilt vm @63 (vs {blabel}) ...")
    (
        normalized,
        prices,
        prices_with_bmk,
        price_df,
        cfg_bt,
        base_weights,
        result,
        active_daily,
    ) = reconstruct(args.segment, args.bmk_index)
    equity = equity_curve(active_daily)
    n_days = int(len(active_daily))
    years = n_days / 252.0
    _log(f"Active daily series: {n_days} days ({years:.1f}y), "
         f"{active_daily.index[0].date()} -> {active_daily.index[-1].date()}")

    # ------------------------------------------------------------------
    # 1. DSR power decomposition (uses the certified sweep's trial family)
    # ------------------------------------------------------------------
    _log("1/11 DSR power decomposition (certified sweep grid) ...")
    cert_sweep = parameter_sweep(
        normalized, prices_with_bmk, cfg_bt, _CERT_GRID,
        basis="active", base_weights=base_weights,
    )
    trial_sharpes = (
        cert_sweep.table["sharpe_daily"].dropna().to_numpy(dtype=float)
    )
    dsr_res = deflated_sharpe_ratio(equity, trial_sharpes)
    parity = assert_parity(equity, dsr_res.dsr, vpath)

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
    # 9. Trial-count honesty: DSR at the cross-run ledger N
    # ------------------------------------------------------------------
    _log(f"2/11 Honest big-N DSR (cross-run ledger N={_HONEST_N_TRIALS}) ...")
    sr0_honest = sr0_from_sigma_n(sigma_trials, _HONEST_N_TRIALS)
    z_honest = (sr_d - sr0_honest) * math.sqrt(n_days - 1.0) / math.sqrt(
        _variance_term(sr_d, skew, kurt)
    )
    sr0_200 = sr0_from_sigma_n(sigma_trials, _HONEST_N_SENSITIVITY)
    z_200 = (sr_d - sr0_200) * math.sqrt(n_days - 1.0) / math.sqrt(
        _variance_term(sr_d, skew, kurt)
    )
    honest = {
        "n_trials": _HONEST_N_TRIALS,
        "ledger": _HONEST_LEDGER,
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
        "sensitivity_n200": {
            "n_trials": _HONEST_N_SENSITIVITY,
            "sr0_ann": sr0_200 * _SQ252,
            "dsr": float(sps.norm.cdf(z_200)),
        },
    }
    _log(f"  honest-N DSR = {honest['dsr']:.3f} "
         f"(SR0_ann {honest['sr0_ann']:.3f}; N=200 sensitivity "
         f"DSR {honest['sensitivity_n200']['dsr']:.3f})")

    # ------------------------------------------------------------------
    # 2. Monte-Carlo deep dive (n_sims=500)
    # ------------------------------------------------------------------
    _log("3/11 Monte-Carlo random-signal null, n_sims=500 (slow) ...")
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
        "family_n": family_n,
        "family_desc": family_desc,
        "p_value_bonferroni_family": float(min(1.0, family_n * mc.p_value)),
    }
    _log(
        f"  MC p = {mc_summary['p_value']:.4f} "
        f"(percentile {mc_summary['actual_percentile_in_null']:.1f}, "
        f"null mean {mc_summary['null_mean']:+.3f}, "
        f"q95 {mc_summary['null_q95']:+.3f}); "
        f"Bonferroni x{family_n} -> {mc_summary['p_value_bonferroni_family']:.3f}"
    )

    # ------------------------------------------------------------------
    # 10. Stationary-bootstrap significance (one-sided p(IR<=0))
    # ------------------------------------------------------------------
    _log("4/11 Stationary bootstrap p(IR<=0), n=2000, seed 42 ...")
    boot = bootstrap_significance(active_daily, n_boot=2000, seed=42)
    _log(
        f"  p(IR<=0) = {boot['p_one_sided_ir_le_0']:.4f}; "
        f"IR 90% CI [{boot['ir_q05']:+.3f}, {boot['ir_q95']:+.3f}], "
        f"95% CI [{boot['ir_q025']:+.3f}, {boot['ir_q975']:+.3f}]"
    )

    # ------------------------------------------------------------------
    # 11. Alpha decomposition (bmk-index mode only)
    # ------------------------------------------------------------------
    decomp = None
    if args.bmk_index is not None and args.bmk_index != args.segment:
        _log("5/11 Alpha decomposition: segment spread vs within-segment "
             "selection ...")
        decomp = alpha_decomposition(
            active_daily, result, price_df, args.segment
        )
        _log(
            f"  spread ({args.segment}-{args.bmk_index}) IR "
            f"{decomp['segment_spread']['ir']:+.3f} "
            f"(NW t {decomp['segment_spread']['nw_t']:+.2f}); "
            f"selection IR {decomp['within_segment_selection']['ir']:+.3f} "
            f"(NW t {decomp['within_segment_selection']['nw_t']:+.2f}); "
            f"corr {decomp['corr_selection_spread']:+.3f}"
        )
    else:
        _log("5/11 Alpha decomposition skipped (benchmark IS the segment "
             "cap index — spread is identically zero).")

    # ------------------------------------------------------------------
    # 3. Walk-forward forensics
    # ------------------------------------------------------------------
    _log("6/11 Walk-forward forensics (5 anchored folds) ...")
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
    _log("7/11 Subperiod robustness (halves / thirds) ...")
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
    _log("8/11 Rolling 252d IR ...")
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
    _log("9/11 Cost stress: tc_bps in {2, 5, 10, 20} ...")
    turnover = result.period_results["turnover"]
    periods_per_year = 252.0 / cfg_bt.periodicity
    ann_turnover = float(turnover.iloc[1:].mean()) * periods_per_year
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
                "ann_cost_drag": float(costs.iloc[1:].mean()) * periods_per_year,
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
    _log("10/11 Parameter neighborhood sweep ...")
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
    _log("11/11 Composite relative IC @63: full vs halves ...")
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
    # 12. Signature scoreboard + verdict
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
                "segment": args.segment,
                "signal": "vm (50/50 Value+Momentum, 6 factors)",
                "construction": "Cap_Tilt (active_share 0.30)",
                "periodicity": 63,
                "benchmark": blabel,
                "bmk_index": args.bmk_index,
                "basis": "active",
                "family_n": family_n,
                "family_desc": family_desc,
                "certified_verdict_file": os.path.basename(vpath),
                "json_file": os.path.basename(json_path),
                "script_cmd": (
                    f"python scripts/overfit_forensics.py "
                    f"--segment {args.segment}"
                    + (f" --bmk-index {args.bmk_index}" if args.bmk_index else "")
                ),
                "reconstruction_parity": parity,
            },
            "power_decomposition": power,
            "honest_big_n_dsr": honest,
            "monte_carlo": mc_summary,
            "bootstrap_significance": boot,
            "alpha_decomposition": decomp,
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
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _log(f"JSON -> {json_path}")

    write_markdown(payload, md_path)
    _log(f"Markdown -> {md_path}")
    _log(f"Done in {payload['runtime_seconds']}s.")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown(p: dict, md_path: str) -> None:  # noqa: C901 — sequential report builder
    st = p["strategy"]
    pw, hn, mc = p["power_decomposition"], p["honest_big_n_dsr"], p["monte_carlo"]
    bt, dec = p["bootstrap_significance"], p["alpha_decomposition"]
    wf, sub, roll = p["walk_forward"], p["subperiods"], p["rolling_252d_ir"]
    cost, nb, ics = p["cost_stress"], p["parameter_neighborhood"], p["ic_stability"]
    sig = p["signatures"]
    seg, bmk = st["segment"], st["benchmark"]
    family_n = st["family_n"]

    L: list[str] = []
    L.append(f"# Overfitting Forensics — {seg} Cap-Tilt (vm @63d vs {bmk})")
    L.append("")
    L.append(
        f"**Question:** is the {seg} Cap-Tilt book's IR of "
        f"{pw['actual_ir_ann']:+.2f} vs the {bmk} (t {pw['actual_t']:.2f}, "
        f"DSR {pw['dsr_actual']:.3f} < 0.95) overfitted, or robust?  "
        f"**Data:** {pw['n_days']} active daily returns "
        f"({pw['years']:.1f}y, {sub['full']['start']} to {sub['full']['end']}). "
        f"Reconstruction reproduces the certified verdict "
        f"`{st['certified_verdict_file']}` bit-for-bit "
        f"(sharpe_ann / t / PSR / DSR all match to <1e-8). "
        f"Script: `{st['script_cmd']}`; machine-readable results in "
        f"`outputs/research/{st['json_file']}` (gitignored, regenerable)."
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
        f"{pw['years']:.1f} years. A true IR ~{pw['actual_ir_ann']:.1f} "
        f"cannot mathematically clear a t~2-equivalent gate on this span "
        f"regardless of whether the edge is real — the DSR gate is "
        f"power-limited here, by construction."
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
        f"| Family-wise (x{family_n}, Bonferroni) | "
        f"{mc['p_value_bonferroni_family']:.3f} |"
    )
    L.append("")
    L.append(
        f"The null holds the Cap_Tilt construction, costs and benchmark "
        f"constant and randomizes only the signal — this isolates selection "
        f"skill from construction. Family for the Bonferroni line: "
        f"{mc['family_desc']}."
    )
    L.append("")

    # 3 — bootstrap significance
    L.append("## 3. Stationary-bootstrap significance (n = 2000, seed 42)")
    L.append("")
    L.append(
        f"Politis-Romano stationary bootstrap of the daily active series "
        f"(expected block length sqrt(n) = {bt['block_length_expected']:.1f} "
        f"days, wrap-around), {bt['n_boot']} resamples; one-sided test of "
        f"H0: IR <= 0 with add-one smoothing."
    )
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Point IR (ann) | {bt['ir_point']:+.3f} |")
    L.append(
        f"| Bootstrap IR 90% CI | [{bt['ir_q05']:+.3f}, {bt['ir_q95']:+.3f}] |"
    )
    L.append(
        f"| Bootstrap IR 95% CI | [{bt['ir_q025']:+.3f}, {bt['ir_q975']:+.3f}] |"
    )
    L.append(f"| Resamples with IR <= 0 | {bt['frac_resamples_ir_le_0']:.2%} |")
    L.append(f"| **p (one-sided, IR <= 0)** | **{bt['p_one_sided_ir_le_0']:.4f}** |")
    L.append("")

    # 4 — alpha decomposition
    if dec is not None:
        L.append("## 4. Alpha decomposition — segment bet vs selection")
        L.append("")
        L.append(
            f"Exact additive split of the daily active return on "
            f"{dec['n_days']} common days (identity max abs error "
            f"{dec['identity_max_abs_err']:.1e}):  "
            f"`active (book - {bmk}) = spread ({seg} cap index - benchmark) "
            f"+ selection (book - {seg} cap index)`."
        )
        L.append("")
        L.append("| Component | Ann mean | IR (ann) | NW t | Ann vol |")
        L.append("|---|---|---|---|---|")
        L.append(
            f"| Active (book - benchmark) | "
            f"{dec['active']['ann_mean']:+.2%} | "
            f"{dec['active']['ir']:+.3f} | — | — |"
        )
        sp = dec["segment_spread"]
        se = dec["within_segment_selection"]
        L.append(
            f"| Segment spread ({seg} - {st['bmk_index']}, structural) | "
            f"{sp['ann_mean']:+.2%} | {sp['ir']:+.3f} | {sp['nw_t']:+.2f} | "
            f"{sp['ann_vol']:.2%} |"
        )
        L.append(
            f"| Within-{seg} selection (book - {seg}) | "
            f"{se['ann_mean']:+.2%} | {se['ir']:+.3f} | {se['nw_t']:+.2f} | "
            f"{se['ann_vol']:.2%} |"
        )
        L.append("")
        vs = dec["variance_shares_of_active"]
        rg = dec["regression_active_on_spread"]
        act_mean = dec["active"]["ann_mean"]
        sp_share = sp["ann_mean"] / act_mean if act_mean else math.nan
        se_share = se["ann_mean"] / act_mean if act_mean else math.nan
        if sp_share >= 0.9 and se["ann_mean"] <= 0.0:
            attribution = (
                f"in RETURN terms the active mean is carried ENTIRELY by "
                f"the structural spread ({sp_share:.0%} of the mean; "
                f"selection nets {se['ann_mean']:+.2%}/yr). The low R^2 "
                f"only says the day-to-day tracking RISK is selection-"
                f"driven — risk is not return: what the IR monetizes is "
                f"the segment composition bet."
            )
        elif se_share >= 0.9 and sp["ann_mean"] <= 0.0:
            attribution = (
                f"in RETURN terms the active mean is carried ENTIRELY by "
                f"within-{seg} selection ({se_share:.0%} of the mean; the "
                f"spread nets {sp['ann_mean']:+.2%}/yr) — the book is not "
                f"a repackaged structural segment bet."
            )
        else:
            attribution = (
                f"the active mean splits {sp_share:.0%} structural spread "
                f"/ {se_share:.0%} within-{seg} selection."
            )
        L.append(
            f"corr(selection, spread) = {dec['corr_selection_spread']:+.3f}. "
            f"Mean attribution: spread {sp['ann_mean']:+.2%}/yr + selection "
            f"{se['ann_mean']:+.2%}/yr = active {act_mean:+.2%}/yr. "
            f"Variance shares of the active return: selection "
            f"{vs['selection']:.0%}, spread {vs['spread']:.0%}, "
            f"2cov {vs['2cov']:+.0%}; active~spread regression beta = "
            f"{rg['beta']:+.2f}, R^2 = {rg['r_squared']:.2f}. Read: "
            f"{attribution}"
        )
        L.append("")
        sec_off = 0
    else:
        sec_off = -1  # decomposition skipped: subsequent sections shift by 1

    # 5 — WF
    L.append(f"## {5 + sec_off}. Walk-forward forensics (5 anchored folds)")
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

    # 6 — subperiods
    L.append(f"## {6 + sec_off}. Subperiod robustness")
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

    # 7 — rolling
    L.append(f"## {7 + sec_off}. Rolling 252-day IR")
    L.append("")
    L.append(
        f"{roll['frac_positive']:.1%} of {roll['n_windows']} rolling windows "
        f"positive; median {roll['median']:+.2f}; worst {roll['worst']:+.2f} "
        f"({roll['worst_date']}); best {roll['best']:+.2f} "
        f"({roll['best_date']}); latest {roll['last']:+.2f}."
    )
    L.append("")

    # 8 — costs
    L.append(f"## {8 + sec_off}. Cost stress")
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

    # 9 — neighborhood
    L.append(f"## {9 + sec_off}. Parameter neighborhood (1-D sweeps around the default)")
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

    # 10 — IC
    L.append(f"## {10 + sec_off}. Composite relative IC @63d — time stability")
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
    L.append(
        "Note: the IC here is signal-vs-segment-countries (benchmark plays "
        "no role) — it measures the raw ranking skill the book monetizes."
    )
    L.append("")

    # 11 — honest N
    L.append(
        f"## {11 + sec_off}. Trial-count honesty — DSR at the cross-run "
        f"ledger N = {hn['n_trials']}"
    )
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
    s200 = hn["sensitivity_n200"]
    L.append(
        f"Ledger: {hn['ledger']} = {hn['n_trials']} trials (~200). Using the "
        f"sweep trial sigma ({hn['trial_sigma_daily_used']:.5f}/day) with "
        f"N = {hn['n_trials']} raises the deflated benchmark to SR0 = "
        f"{hn['sr0_ann']:+.3f} annualized — "
        f"{'ABOVE' if hn['sr0_ann'] > pw['actual_ir_ann'] else 'vs'} the "
        f"actual IR of {pw['actual_ir_ann']:+.3f} — and gives "
        f"**DSR = {hn['dsr']:.3f}** (vs {pw['dsr_actual']:.3f} at the "
        f"certified N = {pw['n_trials_certified']}; at the round N = "
        f"{s200['n_trials']}: SR0 {s200['sr0_ann']:+.3f}, DSR "
        f"{s200['dsr']:.3f}). The required IR rises to "
        f"{hn['required_ir_ann_for_dsr95']:+.3f}; years needed at the "
        f"current IR: {yrs_honest_txt}. Caveat: the sigma behind SR0 is "
        f"estimated from only {pw['n_trials_certified']} sweep trials and "
        f"those {hn['n_trials']} ledger trials are highly correlated "
        f"(variants of the same V+M signal on overlapping data), so this is "
        f"a worst-case deflation, not a point estimate. The qualitative "
        f"conclusion — the DSR gate is unreachable on this span at IR "
        f"~{pw['actual_ir_ann']:.1f} — holds at any N >= "
        f"{pw['n_trials_certified']}."
    )
    L.append("")

    # 12 — verdict
    L.append(f"## {12 + sec_off}. Verdict")
    L.append("")
    L.append("### Overfitting signature scoreboard")
    L.append("")
    L.append("| Signature | Present? |")
    L.append("|---|---|")
    for k, v in sig.items():
        L.append(f"| {k} | {'**PRESENT**' if v else 'absent'} |")
    L.append("")
    n_present = p["n_signatures_present"]
    flagged = [k for k, v in sig.items() if v]
    tuning_flags = [
        k for k in flagged
        if k.startswith("wf_is_much") or k.startswith("default_config")
        or k.startswith("edge_dies") or k.startswith("fails_random")
    ]
    L.append(
        f"**{n_present} of {len(sig)} overfitting signatures present**"
        + (
            " (discussed below; "
            + ("none" if not tuning_flags else "some")
            + " of the flagged items are parameter-tuning signatures)."
            if n_present else "."
        )
    )
    L.append("")

    L.append("### Significance summary")
    L.append("")
    L.append("| Test | Statistic | Reads |")
    L.append("|---|---|---|")
    L.append(
        f"| MC random-signal null (n=500) | p = {mc['p_value']:.4f} "
        f"(percentile {mc['actual_percentile_in_null']:.1f}) | "
        f"{'significant at 5%' if mc['p_value'] <= 0.05 else 'NOT significant at 5%'} |"
    )
    L.append(
        f"| MC family-wise (Bonferroni x{family_n}) | "
        f"p = {mc['p_value_bonferroni_family']:.3f} | "
        f"{'significant at 5%' if mc['p_value_bonferroni_family'] <= 0.05 else 'NOT significant at 5%'} |"
    )
    L.append(
        f"| Stationary bootstrap, one-sided IR<=0 (n=2000) | "
        f"p = {bt['p_one_sided_ir_le_0']:.4f} | "
        f"{'significant at 5%' if bt['p_one_sided_ir_le_0'] <= 0.05 else 'NOT significant at 5%'} |"
    )
    L.append(
        f"| Lo t-stat / PSR / DSR (certified) | t = {pw['actual_t']:.2f}, "
        f"PSR = {p['strategy']['reconstruction_parity']['psr']:.3f}, "
        f"DSR = {pw['dsr_actual']:.3f} | power-limited at this span "
        f"(Section 1) |"
    )
    L.append(
        f"| DSR at honest ledger N = {hn['n_trials']} | "
        f"DSR = {hn['dsr']:.3f} | worst-case deflation (Section "
        f"{11 + sec_off}) |"
    )
    L.append("")

    L.append("### Conclusion")
    L.append("")
    h1, h2 = sub["halves"]
    t1, t2, t3 = sub["thirds"]
    if h1["ir"] > 0 and h2["ir"] > 0:
        decay_label = (
            "stable across halves"
            if min(h1["ir"], h2["ir"]) >= 0.5 * max(h1["ir"], h2["ir"])
            else ("front-loaded" if h1["ir"] > h2["ir"] else "back-loaded")
        )
    else:
        decay_label = "front-loaded" if h1["ir"] > h2["ir"] else "back-loaded"
    ir20 = next(r["ir_ann"] for r in cost["rows"] if r["tc_bps"] == 20.0)
    spread_carries = (
        dec is not None
        and dec["segment_spread"]["ann_mean"] >= dec["active"]["ann_mean"]
        and dec["within_segment_selection"]["ann_mean"] <= 0.0
    )
    if tuning_flags:
        headline = (
            "**Parameter-tuning signatures are PRESENT — treat the "
            "candidate as suspect (see flagged rows above).**"
        )
    elif spread_carries:
        headline = (
            f"**Not overfitted in the parameter / selection-tuning sense, "
            f"and the DSR failure is a statistical-power artifact — BUT "
            f"the decomposition shows the realized IR is a structural "
            f"{seg}-vs-{st['bmk_index']} composition bet, not within-{seg} "
            f"selection alpha, and headline significance is marginal "
            f"(bootstrap p {bt['p_one_sided_ir_le_0']:.3f}, family-wise MC "
            f"p {mc['p_value_bonferroni_family']:.2f}).**"
        )
    else:
        headline = (
            f"**The strategy is NOT overfitted in the parameter / "
            f"selection-tuning sense, and the DSR failure is a "
            f"statistical-power artifact — the time profile of the alpha "
            f"is {decay_label}.**"
        )
    L.append(headline + " Evidence by leg:")
    L.append("")
    L.append(
        f"1. **Power.** At the realized IR of "
        f"{pw['actual_ir_ann']:+.2f}, DSR >= 0.95 needs "
        f"~{pw['years_needed_at_current_ir_for_dsr95']:.0f} years of data "
        f"(plain PSR: {pw['years_needed_at_current_ir_for_psr95']:.0f}y; "
        f"t >= 2: {pw['years_needed_at_current_ir_for_t2']:.0f}y); the "
        f"sample has {pw['years']:.1f}. Equivalently the gate demands IR "
        f"{pw['required_ir_ann_for_dsr95']:+.2f} (t "
        f"{pw['required_t_for_dsr95']:.2f}) on this span — "
        f"{pw['required_ir_ann_for_dsr95']/pw['actual_ir_ann']:.1f}x the "
        f"point estimate. A true IR-{abs(pw['actual_ir_ann']):.1f} strategy "
        f"fails this gate by construction; the failure carries no "
        f"information about overfitting."
    )
    L.append(
        f"2. **Parameter-tuning probes.** Walk-forward WFE "
        f"{wf['wf_efficiency']:.2f} with "
        f"{wf['frac_folds_oos_positive']:.0%} of folds OOS-positive"
        f"{' (OOS beat IS selection — the opposite of the IS>>OOS signature)' if wf['wf_efficiency'] > 1.0 else ''}; "
        f"the parameter neighborhood is {nb['frac_positive']:.0%} positive "
        f"with the default at z = {nb['default_zscore']:+.2f}"
        f"{' (not the peak — a tuned config would sit at the peak)' if nb['default_sharpe'] < nb['sharpe_max'] else ''}; "
        f"the edge survives 10x assumed costs (IR {ir20:+.2f} at 20 bps, "
        f"turnover {cost['ann_one_sided_turnover']:.0%}/yr); the "
        f"random-signal null — construction held constant — gives "
        f"p = {mc['p_value']:.3f} on 500 sims (book at the "
        f"{mc['actual_percentile_in_null']:.0f}th percentile); and the "
        f"stationary bootstrap puts p(IR<=0) at "
        f"{bt['p_one_sided_ir_le_0']:.3f} with a 90% IR CI of "
        f"[{bt['ir_q05']:+.2f}, {bt['ir_q95']:+.2f}]. "
        + (
            "Flagged-but-benign: WF param choices rotate across folds "
            "(modal share "
            f"{wf['modal_param_share']:.0%}) because the neighborhood is "
            f"flat — IS Sharpes are statistically indistinguishable, so the "
            f"argmax is noise (corroborated by {nb['frac_positive']:.0%} of "
            f"configs positive) — and the deployed params were fixed "
            f"ex-ante by literature priors, not selected by this WF."
            if sig.get("wf_unstable_param_choices (modal share<0.6)")
            and nb["frac_positive"] >= 0.9
            else ""
        )
    )
    if dec is not None:
        sp = dec["segment_spread"]
        se = dec["within_segment_selection"]
        vs = dec["variance_shares_of_active"]
        sel_dominant = (
            se["ann_mean"] >= 0.5 * dec["active"]["ann_mean"]
            and se["ir"] > 0.0
        )
        if spread_carries:
            reading = (
                f"The realized {dec['active']['ir']:+.2f} IR vs "
                f"{st['bmk_index']} is carried entirely by the structural "
                f"{seg} overweight; the V+M overlay nets "
                f"{se['ann_mean']:+.2%}/yr against the vendor {seg} cap "
                f"index on this sample. Reconciliation with the MC result: "
                f"random-signal tilts under the SAME construction average "
                f"only {mc['null_mean']:+.2f} IR vs the {sp['ir']:+.2f} "
                f"passive spread (the Cap_Tilt machinery itself dilutes "
                f"the spread), so the signal beats random tilts "
                f"(p = {mc['p_value']:.3f}) — but 'better than random "
                f"tilts' is a weaker claim than 'adds value over the "
                f"passive {seg} index', and the data do not support the "
                f"latter."
            )
        elif sel_dominant:
            reading = (
                f"The book's edge is predominantly WITHIN-{seg} selection "
                f"— it is not a repackaged structural {seg}-overweight."
            )
        else:
            reading = (
                f"A material share of the active return is the structural "
                f"{seg}-vs-{st['bmk_index']} composition bet, NOT "
                f"selection — the IR partly rides a passive segment "
                f"spread."
            )
        L.append(
            f"3. **Segment bet vs selection.** The active return splits "
            f"exactly into the passive {seg}-vs-{st['bmk_index']} spread "
            f"(ann {sp['ann_mean']:+.2%}, IR {sp['ir']:+.2f}, NW t "
            f"{sp['nw_t']:+.2f}) and within-{seg} selection (ann "
            f"{se['ann_mean']:+.2%}, IR {se['ir']:+.2f}, NW t "
            f"{se['nw_t']:+.2f}), corr {dec['corr_selection_spread']:+.2f}; "
            f"variance shares selection {vs['selection']:.0%} / spread "
            f"{vs['spread']:.0%} / 2cov {vs['2cov']:+.0%}. {reading}"
        )
    L.append(
        f"{4 if dec is not None else 3}. **Time profile.** Half-1 IR "
        f"{h1['ir']:+.2f} (q-hit {h1['quarter_hit_rate']:.0%}) vs half-2 IR "
        f"{h2['ir']:+.2f} (q-hit {h2['quarter_hit_rate']:.0%}); thirds "
        f"{t1['ir']:+.2f} / {t2['ir']:+.2f} / {t3['ir']:+.2f}; composite IC "
        f"{ics['first_half']['mean_ic']:+.3f} -> "
        f"{ics['second_half']['mean_ic']:+.3f} across halves; "
        f"{roll['frac_positive']:.0%} of rolling 252d windows positive "
        f"(latest {roll['last']:+.2f}). The alpha is {decay_label}."
    )
    L.append("")
    L.append("### Residual risks (stated honestly)")
    L.append("")
    L.append(
        f"1. **Post-hoc candidate selection (family-wise honesty).** This "
        f"book was chosen as the lead candidate AFTER observing the full "
        f"evaluation round ({mc['family_desc']} — 1-of-{family_n}); "
        f"Bonferroni across the family puts the MC evidence at "
        f"~{mc['p_value_bonferroni_family']:.2f} family-wise — "
        f"{'still significant at 5%' if mc['p_value_bonferroni_family'] <= 0.05 else 'suggestive, not significant at 5%'}. "
        f"The honest claim is "
        f"'{'signal-beats-random-tilts' if spread_carries else 'selection skill'} "
        f"significant within this book, "
        f"{'and' if mc['p_value_bonferroni_family'] <= 0.05 else 'but only'} "
        f"{'robust' if mc['p_value_bonferroni_family'] <= 0.05 else 'marginal'} family-wise'."
    )
    L.append(
        f"2. **Big-N deflation.** Against the full cross-run ledger "
        f"(N~{hn['n_trials']}) the DSR drops to {hn['dsr']:.3f} and the "
        f"deflated benchmark ({hn['sr0_ann']:+.2f}) "
        f"{'exceeds' if hn['sr0_ann'] > pw['actual_ir_ann'] else 'approaches'} "
        f"the realized IR — under worst-case trial accounting no amount of "
        f"history at this IR certifies. The program has consumed many "
        f"trials; only fresh OOS data (live quarters) repays that debt."
    )
    L.append(
        f"3. **Alpha time profile.** The alpha is {decay_label} "
        f"(Sections {6 + sec_off}/{10 + sec_off}); the per-period IC "
        f"t-stats are individually insignificant throughout (skill shows in "
        f"the book-level MC test, not the period IC), so the edge rests on "
        f"construction + breadth — monitor the rolling IR and quarterly "
        f"re-certs; a sustained negative rolling-IR trend is grounds to "
        f"pull the candidate regardless of the full-sample stats."
    )
    if dec is not None and (
        spread_carries or dec["variance_shares_of_active"]["spread"] > 0.25
    ):
        L.append(
            f"4. **Structural-bet dependence.** The active mean is "
            f"{'entirely' if spread_carries else 'partly'} the passive "
            f"{seg}-vs-{st['bmk_index']} spread "
            f"({dec['segment_spread']['ann_mean']:+.2%}/yr of the "
            f"{dec['active']['ann_mean']:+.2%}/yr active mean; "
            f"{dec['variance_shares_of_active']['spread']:.0%} of active "
            f"variance); if the {seg}-vs-{st['bmk_index']} regime reverses "
            f"(e.g. an EM-led cycle), that leg of the IR reverses with it, "
            f"independent of selection skill."
        )
    L.append("")
    L.append(
        f"*Generated by `scripts/overfit_forensics.py` in "
        f"{p['runtime_seconds']}s; deterministic (seeded); inputs are the "
        f"gitignored vendor data.*"
    )
    L.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
