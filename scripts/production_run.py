"""Production pipeline: deployed-strategy registry -> periodic artifacts.

ONE reproducible command, runnable periodically (e.g. quarterly after an
``Inputs/`` refresh), producing per-strategy artifacts for every deployed
strategy in ``configs/production.json``:

==========================  ===================================================
artifact                    content
==========================  ===================================================
allocations.csv             historical_weights — every rebalance (dates x
                            countries; Cap_Tilt active book rows sum to 1)
allocations_latest.json     latest weights dict + rebalance date + next
                            expected rebalance date (last grid date +
                            ``periodicity`` business days)
metrics.json                mandate stats (daily, ppy=252, vs benchmark),
                            period stats, annualized turnover, composite IC
                            (absolute + relative), last-252d active return /
                            IR, terminal equity value; echo of the registry
                            entry + data end date + git commit hash.
                            TIMESTAMP-FREE by design (determinism: the run
                            timestamp lives only in manifest.json).
signal_latest.json          per country at the last score date: normalized
                            composite score, per-category rebased
                            contribution, score change over the trailing
                            ``periodicity`` index rows, current weight,
                            base (cap) weight
signal_history.csv          normalized composite score per country per date
                            (full daily history)
signal_history_monthly.csv  month-end sample of signal_history (lighter,
                            for the dashboard)
contributions_latest.csv    per-category x country rebased contributions at
                            the last score date
tca.json                    transaction-cost analysis: turnover summary, top
                            traded countries, cost layers in annualized bps,
                            cumulative layer IRs (gross -> net spread ->
                            +expense -> +mgmt), flat one-way breakeven bps,
                            cost-model echo (configs/costs.json)
turnover.csv                per-rebalance one-way turnover + per-period
                            spread/commission/expense/mgmt cost columns from
                            the TCA decomposition + gross active return
==========================  ===================================================

A ``manifest.json`` at the run root records the run timestamp, data end
date, strategies produced, file inventory with row counts, package version
and git commit hash.

Determinism contract: same inputs -> identical artifact bytes (no random
anywhere on the ``--quick`` path; the full validation suite is seeded).
The run directory is stamped with the DATA END date
(``outputs/production/run_{YYYYMMDD}/``), so re-running on unchanged inputs
overwrites the same folder with identical content.

Usage
-----
    python scripts/production_run.py [--config configs/default.json]
        [--registry configs/production.json]
        [--output-dir outputs/production]
        [--strategy ID] [--quick]

``--quick`` skips the validation scorecard: only engine + signal artifacts
are produced (``metrics.json`` carries ``"validation": null``).

Note: requires local data (``Inputs/``, ``Classification.xlsx``) which is
gitignored — present on the research machine only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone

# Allow running from repo root without installing the package; the scripts
# dir itself is added so research_run can be imported as a module.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

_DEFAULT_CONFIG = "configs/default.json"
_DEFAULT_REGISTRY = "configs/production.json"
_DEFAULT_OUTPUT_DIR = "outputs/production"
_DEFAULT_COSTS = os.path.join(_REPO_ROOT, "configs", "costs.json")

_MIN_COUNTRIES = 8

#: Registry fields every deployed strategy MUST declare (bmk_index may be
#: null = use the segment's own vendor index column).
_REQUIRED_FIELDS = (
    "id", "label", "segment", "prior_set", "periodicity",
    "construction", "active_share", "bmk_index",
)

#: Full-suite validation knobs (mirror research_run non-quick defaults).
_N_BOOT = 500
_N_MC = 100
_N_FOLDS = 5
_SEED = 42
_GRID = {"relative_selection_score": (3, 5, 7), "periodicity": (21, 63)}


def _log(msg: str) -> None:
    print(f"[production_run] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def git_commit_hash(repo_root: str = _REPO_ROOT) -> str:
    """Current git commit hash, or 'unknown' (never raises)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_root, check=False,
        )
        commit = out.stdout.strip()
        return commit if out.returncode == 0 and commit else "unknown"
    except OSError:
        return "unknown"


def package_version() -> str:
    """Installed package version, or 'unknown' (never raises)."""
    try:
        from importlib.metadata import version

        return version("country-rotation-strategy")
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return "unknown"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def load_registry(path) -> dict:
    """Load and schema-check the deployed-strategy registry JSON."""
    with open(path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    strategies = registry.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise SystemExit(
            f"[production_run] ERROR: registry '{path}' has no non-empty "
            f"'strategies' list."
        )
    seen: set = set()
    for entry in strategies:
        missing = [k for k in _REQUIRED_FIELDS if k not in entry]
        if missing:
            raise SystemExit(
                f"[production_run] ERROR: registry entry "
                f"{entry.get('id', '<no id>')!r} is missing required "
                f"field(s): {missing}."
            )
        if entry["id"] in seen:
            raise SystemExit(
                f"[production_run] ERROR: duplicate strategy id "
                f"{entry['id']!r} in registry."
            )
        seen.add(entry["id"])
    return registry


# ---------------------------------------------------------------------------
# Per-strategy book construction (research_run reuse)
# ---------------------------------------------------------------------------

def build_strategy_inputs(processed, classification, strategy_cfg: dict) -> dict:
    """Universe, benchmark-extended prices, composite and cap base weights
    for one registry strategy — pure research_run reuse, no engine run.

    ``bmk_index`` null -> the segment's own vendor index column from the
    Price input (``research_run --bmk-source index`` equivalent); a name
    (e.g. 'World' = ACWI equivalent) pins that cross-segment vendor column
    instead (``--bmk-index`` equivalent).
    """
    import research_run

    segment = strategy_cfg["segment"]
    if "Price" not in processed:
        raise SystemExit(
            "[production_run] ERROR: 'Price' metric missing from inputs."
        )
    price_df = processed["Price"]

    universe_all = research_run.extract_universe(classification, segment)
    universe = [c for c in universe_all if c in price_df.columns]
    if len(universe) < _MIN_COUNTRIES:
        raise SystemExit(
            f"[production_run] ERROR: segment '{segment}' has only "
            f"{len(universe)} countries with prices (< {_MIN_COUNTRIES})."
        )

    bmk_col = strategy_cfg.get("bmk_index") or segment
    if bmk_col not in price_df.columns:
        raise SystemExit(
            f"[production_run] ERROR: vendor index column '{bmk_col}' "
            f"missing from the Price input (required as the "
            f"'{strategy_cfg['id']}' benchmark)."
        )
    prices = price_df[universe].copy()
    prices_with_bmk = prices.copy()
    prices_with_bmk[bmk_col] = price_df[bmk_col]

    factor_scores, _ = research_run.build_factor_scores(processed, universe)
    factor_set = research_run.select_prior_factors(
        factor_scores, strategy_cfg.get("prior_set", "vm")
    )
    if not factor_set:
        raise SystemExit(
            f"[production_run] ERROR: no '{strategy_cfg.get('prior_set')}' "
            f"prior factors available for segment '{segment}'."
        )
    normalized, contributions, category_weights = research_run.build_composite(
        factor_scores, tuple(factor_set)
    )

    base_weights = None
    if strategy_cfg["construction"] == "cap_tilt":
        if "Market_Cap" not in processed:
            raise SystemExit(
                "[production_run] ERROR: construction 'cap_tilt' requires "
                "the 'Market_Cap' metric (Inputs/Market_Cap.xlsx)."
            )
        mcap_df = processed["Market_Cap"]
        mcap_cols = [c for c in universe if c in mcap_df.columns]
        if len(mcap_cols) < _MIN_COUNTRIES:
            raise SystemExit(
                f"[production_run] ERROR: Market_Cap covers only "
                f"{len(mcap_cols)} universe countries (< {_MIN_COUNTRIES})."
            )
        mcap = mcap_df[mcap_cols].ffill()
        mcap = mcap[mcap.sum(axis=1) > 0]
        base_weights = mcap.div(mcap.sum(axis=1), axis=0)

    return {
        "universe": universe,
        "bmk_col": bmk_col,
        "prices": prices,
        "prices_with_bmk": prices_with_bmk,
        "normalized": normalized,
        "contributions": contributions,
        "category_weights": category_weights,
        "factor_set": factor_set,
        "base_weights": base_weights,
    }


# ---------------------------------------------------------------------------
# Metric blocks
# ---------------------------------------------------------------------------

def _composite_ic_stats(normalized, prices, periodicity: int) -> dict:
    """Full-window composite IC stats, both methods (informational).

    scipy ConstantInputWarning silenced: warm-up composite rows are constant
    0.5 cross-sections whose NaN ICs are dropped anyway (research_run
    precedent in run_screening / prior_factor_evidence).
    """
    from scipy.stats import ConstantInputWarning

    from country_rotation.backtest import ic as ic_module

    out: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for method in ("absolute", "relative"):
            ic_df = ic_module.information_coefficient(
                normalized, prices, periodicity, method
            )
            out[method] = ic_module.ic_stats(ic_df["IC"].dropna())
    return out


def _last_252d_active(dr, db) -> dict:
    """Trailing-252-business-day active block from the daily curves.

    active_return_cum = terminal wealth gap over the window;
    ir_ann = mean(daily active) / std(daily active) * sqrt(252).
    """
    import numpy as np
    import pandas as pd

    aligned = pd.DataFrame({"port": dr, "bmk": db}).dropna()
    window = min(252, len(aligned))
    if window < 2:
        return {"window_days": window, "active_return_cum": None, "ir_ann": None}
    tail = aligned.tail(window)
    active = tail["port"] - tail["bmk"]
    cum = float((1.0 + tail["port"]).prod() - (1.0 + tail["bmk"]).prod())
    std = float(active.std())
    ir = float(active.mean() / std * np.sqrt(252.0)) if std > 0.0 else None
    return {"window_days": int(window), "active_return_cum": cum, "ir_ann": ir}


def _validation_stats(
    normalized, prices_with_bmk, cfg_bt, base_weights, cost_bps=None
) -> dict:
    """Full validation scorecard (slow path; seeded -> deterministic)."""
    from country_rotation.validation.scorecard import compute_validation

    vr = compute_validation(
        normalized,
        prices_with_bmk,
        cfg_bt,
        _GRID,
        n_boot=_N_BOOT,
        n_mc=_N_MC,
        n_folds=_N_FOLDS,
        seed=_SEED,
        basis="active",
        base_weights=base_weights,
        cost_bps=cost_bps,
    )
    return {
        "overall": vr.verdict.overall,
        "checks": vr.verdict.checks,
        "notes": list(vr.verdict.notes),
        "sharpe_ann": vr.sharpe.sharpe_ann,
        "sharpe_t_stat": vr.sharpe.t_stat,
        "psr": vr.psr.psr,
        "dsr": vr.dsr.dsr,
        "mc_p_value": vr.mc.p_value,
        "wf_efficiency": vr.walkforward.wf_efficiency,
        "frac_oos_positive": vr.walkforward.frac_folds_oos_positive,
        "bootstrap_ci": [vr.bootstrap.ci_low, vr.bootstrap.ci_high],
        "nw_t_vs_eqw": vr.nw_vs_eqw.t_stat,
        "stability_frac_positive": vr.stability.frac_positive,
        "stability_default_zscore": vr.stability.default_zscore,
    }


# ---------------------------------------------------------------------------
# Per-strategy artifact production
# ---------------------------------------------------------------------------

def produce_strategy_artifacts(
    processed,
    classification,
    strategy_cfg: dict,
    out_dir: str,
    quick: bool = True,
    base_cfg=None,
    cost_model=None,
) -> dict:
    """Run one registry strategy end-to-end and write its artifact set.

    Pure-ish: everything is derived from ``processed`` + ``classification``
    + ``strategy_cfg`` (+ the platform ``base_cfg`` engine defaults); the
    only side effects are the artifact files under ``out_dir``.  No
    randomness on the quick path; the full validation suite is seeded.

    The engine ALWAYS runs with the per-country cost vector from
    ``cost_model`` (default: ``configs/costs.json``) — weights are
    unchanged (costs never feed selection), only net returns and the
    transaction-cost column move; TCA artifacts (``tca.json`` +
    ``turnover.csv``) and the ``metrics.json`` ``net_of_fee`` block are
    derived from the same model.

    Returns the manifest info block: file inventory with row counts, latest
    rebalance dates and top-5 holdings.
    """
    import pandas as pd
    import research_run

    from country_rotation.backtest import tca
    from country_rotation.backtest.engine import Engine
    from country_rotation.backtest.metrics import summary
    from country_rotation.config import BacktestConfig

    if base_cfg is None:
        base_cfg = BacktestConfig()
    if cost_model is None:
        cost_model = tca.load_cost_model(_DEFAULT_COSTS)

    inputs = build_strategy_inputs(processed, classification, strategy_cfg)
    universe = inputs["universe"]
    normalized = inputs["normalized"]
    contributions = inputs["contributions"]
    base_weights = inputs["base_weights"]
    periodicity = int(strategy_cfg["periodicity"])

    cfg_bt = dataclasses.replace(
        base_cfg,
        bmk=inputs["bmk_col"],
        bmk_weight=0.0,
        mode="active",
        periodicity=periodicity,
        selection_criteria="relative",
        weighting_method=(
            "Cap_Tilt" if strategy_cfg["construction"] == "cap_tilt" else "Equal"
        ),
        active_share=float(strategy_cfg["active_share"]),
    )

    cost_bps = cost_model.cost_vector(universe)
    _log(
        f"[{strategy_cfg['id']}] engine run: {len(universe)} countries, "
        f"bmk='{inputs['bmk_col']}', construction="
        f"{strategy_cfg['construction']}, periodicity={periodicity}, "
        f"costs {cost_bps.min():g}-{cost_bps.max():g} bps one-way ..."
    )
    result = Engine(
        normalized, inputs["prices_with_bmk"], cfg_bt,
        base_weights=base_weights, cost_bps=cost_bps,
    ).run()
    if result.historical_weights.empty:
        raise SystemExit(
            f"[production_run] ERROR: strategy '{strategy_cfg['id']}' "
            f"produced no rebalances (check periodicity vs data span)."
        )

    # --- TCA: turnover + cost decomposition + breakeven ------------------
    turnover_report = tca.turnover_analysis(result)
    cost_report = tca.cost_decomposition(
        result, cost_model, strategy_cfg["segment"]
    )
    try:
        breakeven_bps = tca.breakeven_cost(result, cost_report.gross_ir)
    except ValueError as exc:  # degenerate book: artifact still produced
        warnings.warn(
            f"[production_run] breakeven undefined for "
            f"'{strategy_cfg['id']}': {exc}",
            stacklevel=2,
        )
        breakeven_bps = None
    layer_irs = dict(cost_report.layer_irs)
    net_of_fee = {
        "gross_ir": layer_irs.get("gross"),
        "net_spread_ir": layer_irs.get("net_spread"),
        "net_spread_expense_ir": layer_irs.get("net_spread_expense"),
        "net_mgmt_50_ir": layer_irs.get("net_mgmt_50bps"),
    }

    os.makedirs(out_dir, exist_ok=True)
    files: dict = {}

    # --- allocations.csv: every rebalance (weights indexed by period end) ---
    hw = result.historical_weights
    hw.to_csv(
        os.path.join(out_dir, "allocations.csv"),
        index_label="date", lineterminator="\n",
    )
    files["allocations.csv"] = len(hw)

    # --- allocations_latest.json ---------------------------------------
    last_reb_date = hw.index[-1]
    next_reb_date = last_reb_date + pd.offsets.BDay(periodicity)
    latest_weights = {
        c: float(w)
        for c, w in hw.iloc[-1].sort_index().items()
        if abs(float(w)) > 1e-12
    }
    allocations_latest = research_run._json_safe({
        "strategy_id": strategy_cfg["id"],
        "rebalance_date": str(pd.Timestamp(last_reb_date).date()),
        "next_rebalance_date": str(pd.Timestamp(next_reb_date).date()),
        "weights": latest_weights,
    })
    with open(
        os.path.join(out_dir, "allocations_latest.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(allocations_latest, f, indent=2)
    files["allocations_latest.json"] = len(latest_weights)

    # --- metrics.json (timestamp-free by design) ------------------------
    dr = result.daily_returns
    db = result.daily_bmk_returns
    mandate_stats = (
        summary(dr, db, 252)
        if dr is not None and db is not None and len(dr) > 1
        else {}
    )
    pr = result.period_results
    ppy = 252.0 / periodicity
    period_stats = (
        summary(pr["portfolio_return_net"], pr["bmk_return"], ppy)
        if len(pr) > 1
        else {}
    )
    # fsum + round(…, 10): the engine's per-period turnover scalars carry
    # 1-ULP run-to-run jitter (alignment-dependent numpy reductions in the
    # parity-locked engine) — exact summation plus 1e-10 quantization keeps
    # metrics.json byte-deterministic.
    _to = pr["turnover"].dropna()
    turnover_ann = (
        round(math.fsum(_to) / len(_to) * ppy, 10) if len(_to) else None
    )
    composite_ic = _composite_ic_stats(normalized, inputs["prices"], periodicity)
    last_252d = (
        _last_252d_active(dr, db)
        if dr is not None and db is not None
        else {"window_days": 0, "active_return_cum": None, "ir_ann": None}
    )
    equity_last = (
        float((1.0 + dr.dropna()).prod()) if dr is not None else None
    )
    validation = None
    if not quick:
        _log(f"[{strategy_cfg['id']}] validation suite (n_boot={_N_BOOT}, "
             f"n_mc={_N_MC}, n_folds={_N_FOLDS}) — the slow part ...")
        validation = _validation_stats(
            normalized, inputs["prices_with_bmk"], cfg_bt, base_weights,
            cost_bps=cost_bps,
        )

    data_end = str(pd.Timestamp(inputs["prices"].index.max()).date())
    metrics = research_run._json_safe({
        "strategy": strategy_cfg,
        "data_end": data_end,
        "git_commit": git_commit_hash(),
        "benchmark_column": inputs["bmk_col"],
        "n_countries": len(universe),
        "universe": universe,
        "factor_set": inputs["factor_set"],
        "category_weights": inputs["category_weights"],
        "n_rebalances": len(pr),
        "mandate_stats": mandate_stats,
        "period_stats": period_stats,
        "turnover_ann": turnover_ann,
        "composite_ic": composite_ic,
        "last_252d_active": last_252d,
        "equity_curve_last": equity_last,
        "net_of_fee": net_of_fee,
        "validation": validation,
    })
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    files["metrics.json"] = 1

    # --- signal_latest.json ---------------------------------------------
    last_sig_date = normalized.index[-1]
    score_now = normalized.iloc[-1]
    prev_pos = len(normalized) - 1 - periodicity
    score_prev = normalized.iloc[prev_pos] if prev_pos >= 0 else None
    weights_row = hw.iloc[-1]
    base_row = None
    if base_weights is not None:
        base_hist = base_weights.loc[:last_sig_date]
        if len(base_hist):
            base_row = base_hist.iloc[-1]

    countries_payload: dict = {}
    for country in universe:
        score = score_now.get(country)
        change = (
            score - score_prev.get(country) if score_prev is not None else None
        )
        countries_payload[country] = {
            "score": score,
            "score_change": change,
            "weight": float(weights_row.get(country, 0.0)),
            "base_weight": (
                float(base_row[country])
                if base_row is not None and country in base_row.index
                else None
            ),
            "contributions": {
                category: (
                    df.loc[last_sig_date, country]
                    if country in df.columns and last_sig_date in df.index
                    else None
                )
                for category, df in contributions.items()
            },
        }
    signal_latest = research_run._json_safe({
        "strategy_id": strategy_cfg["id"],
        "date": str(pd.Timestamp(last_sig_date).date()),
        "periodicity": periodicity,
        "countries": countries_payload,
    })
    with open(
        os.path.join(out_dir, "signal_latest.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(signal_latest, f, indent=2)
    files["signal_latest.json"] = len(countries_payload)

    # --- signal_history.csv (full daily) + month-end sample --------------
    normalized.to_csv(
        os.path.join(out_dir, "signal_history.csv"),
        index_label="date", lineterminator="\n",
    )
    files["signal_history.csv"] = len(normalized)

    monthly = normalized[
        ~normalized.index.to_period("M").duplicated(keep="last")
    ]
    monthly.to_csv(
        os.path.join(out_dir, "signal_history_monthly.csv"),
        index_label="date", lineterminator="\n",
    )
    files["signal_history_monthly.csv"] = len(monthly)

    # --- contributions_latest.csv (category x country) -------------------
    contrib_latest = pd.DataFrame(
        {cat: df.loc[last_sig_date] for cat, df in contributions.items()}
    ).T
    contrib_latest.to_csv(
        os.path.join(out_dir, "contributions_latest.csv"),
        index_label="category", lineterminator="\n",
    )
    files["contributions_latest.csv"] = len(contrib_latest)

    # --- ic_series.csv (per-period Spearman IC, both methods) -----------
    from scipy.stats import ConstantInputWarning

    from country_rotation.backtest import ic as ic_module

    ic_parts: list[pd.Series] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for method, col in (("absolute", "ic_absolute"),
                            ("relative", "ic_relative")):
            ic_df = ic_module.information_coefficient(
                normalized, inputs["prices"], periodicity, method
            )
            ic_parts.append(ic_df["IC"].rename(col))

    ic_series_df = pd.concat(ic_parts, axis=1).sort_index()
    ic_series_df.to_csv(
        os.path.join(out_dir, "ic_series.csv"),
        index_label="date", lineterminator="\n",
    )
    files["ic_series.csv"] = len(ic_series_df)

    # --- tca.json (turnover + cost layers + breakeven + model echo) -----
    tca_payload = research_run._json_safe({
        "strategy_id": strategy_cfg["id"],
        "segment": strategy_cfg["segment"],
        "cost_model": {
            "as_of": cost_model.as_of,
            "commission_bps": cost_model.commission_bps,
            "expense_ratio_bps": float(
                cost_model.expense_ratio_bps[strategy_cfg["segment"]]
            ),
            "mgmt_fee_scenarios_bps": list(cost_model.mgmt_fee_scenarios_bps),
            "universe_one_way_bps": {
                c: float(v) for c, v in cost_bps.items()
            },
        },
        "turnover": {
            "ann_one_way": turnover_report.ann_one_way_turnover,
            "avg_per_rebalance": turnover_report.avg_turnover,
            "max_per_rebalance": turnover_report.max_turnover,
            "total_name_changes": turnover_report.total_name_changes,
        },
        "country_avg_traded": {
            c: float(v)
            for c, v in turnover_report.country_avg_traded.items()
        },
        "cost_layers_ann_bps": {
            "spread": cost_report.spread_ann_bps,
            "commission": cost_report.commission_ann_bps,
            "expense": cost_report.expense_ann_bps,
            "mgmt": {
                f"{s:g}bps": v for s, v in cost_report.mgmt_ann_bps.items()
            },
        },
        "layer_irs": layer_irs,
        "net_of_fee": net_of_fee,
        "breakeven_bps": breakeven_bps,
    })
    with open(os.path.join(out_dir, "tca.json"), "w", encoding="utf-8") as f:
        json.dump(tca_payload, f, indent=2)
    files["tca.json"] = 1

    # --- turnover.csv (per-rebalance turnover + TCA cost columns) -------
    turnover_df = pd.concat(
        [
            turnover_report.per_rebalance.rename("turnover"),
            cost_report.per_period,
            pr["active_return"].rename("active_return"),
        ],
        axis=1,
    )
    turnover_df.to_csv(
        os.path.join(out_dir, "turnover.csv"),
        index_label="date", lineterminator="\n",
    )
    files["turnover.csv"] = len(turnover_df)

    top5 = weights_row.sort_values(ascending=False).head(5)
    _log(
        f"[{strategy_cfg['id']}] latest rebalance {allocations_latest['rebalance_date']}"
        f" | top-5: "
        + ", ".join(f"{c} {w:.1%}" for c, w in top5.items())
    )

    return research_run._json_safe({
        "id": strategy_cfg["id"],
        "label": strategy_cfg.get("label"),
        "data_end": data_end,
        "n_countries": len(universe),
        "latest_rebalance": allocations_latest["rebalance_date"],
        "next_rebalance": allocations_latest["next_rebalance_date"],
        "latest_signal_date": signal_latest["date"],
        "top5_weights": {c: float(w) for c, w in top5.items()},
        "files": files,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Production pipeline: run every deployed strategy in the "
            "registry and write per-strategy artifacts (allocations, "
            "metrics, signal snapshot + history) plus a run manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default=_DEFAULT_CONFIG, help="Path to JSON platform config."
    )
    parser.add_argument(
        "--registry", default=_DEFAULT_REGISTRY,
        help="Path to the deployed-strategy registry JSON.",
    )
    parser.add_argument(
        "--output-dir", default=_DEFAULT_OUTPUT_DIR, dest="output_dir",
        help="Root directory for run_{YYYYMMDD}/ artifact folders "
             "(stamped with the DATA END date for reproducibility).",
    )
    parser.add_argument(
        "--strategy", default=None, metavar="ID",
        help="Only produce artifacts for this registry strategy id.",
    )
    parser.add_argument(
        "--quick", action="store_true", default=False,
        help="Skip the validation scorecard: engine + signal artifacts only.",
    )
    args = parser.parse_args()

    t0 = time.time()

    # Lazy imports: keep --help fast and import-side-effect free.
    import pandas as pd
    import research_run

    from country_rotation.config import load_config

    cfg = load_config(args.config)
    registry = load_registry(args.registry)
    strategies = registry["strategies"]
    if args.strategy is not None:
        strategies = [s for s in strategies if s["id"] == args.strategy]
        if not strategies:
            raise SystemExit(
                f"[production_run] ERROR: strategy id {args.strategy!r} not "
                f"in registry (available: "
                f"{[s['id'] for s in registry['strategies']]})."
            )
    _log(f"Registry '{args.registry}': producing "
         f"{len(strategies)} strategy(ies).")

    # Cost model loaded ONCE — every strategy engine runs net of the
    # per-country spread+commission vector (weights unchanged by costs).
    from country_rotation.backtest.tca import load_cost_model

    cost_model = load_cost_model(_DEFAULT_COSTS)
    _log(f"Cost model '{_DEFAULT_COSTS}' (as_of {cost_model.as_of}).")

    # Ingest ONCE — shared across strategies.
    processed, classification = research_run.ingest(cfg)
    if "Price" not in processed:
        raise SystemExit(
            "[production_run] ERROR: 'Price' metric missing from inputs."
        )
    data_end = pd.Timestamp(processed["Price"].index.max())
    run_dir = os.path.join(args.output_dir, f"run_{data_end:%Y%m%d}")
    _log(f"Data end {data_end.date()} -> run dir '{run_dir}'.")

    produced: dict = {}
    for strategy_cfg in strategies:
        _log("=" * 64)
        _log(f"Strategy '{strategy_cfg['id']}' — {strategy_cfg['label']}")
        out_dir = os.path.join(run_dir, strategy_cfg["id"])
        produced[strategy_cfg["id"]] = produce_strategy_artifacts(
            processed,
            classification,
            strategy_cfg,
            out_dir,
            quick=args.quick,
            base_cfg=cfg.backtest,
            cost_model=cost_model,
        )

    manifest = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "data_end": str(data_end.date()),
        "config": args.config,
        "registry": args.registry,
        "registry_note": registry.get("as_of_note"),
        "quick": bool(args.quick),
        "package_version": package_version(),
        "git_commit": git_commit_hash(),
        "strategies": produced,
    }
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _log("=" * 64)
    _log(f"Manifest -> {manifest_path}")
    _log(f"Done: {len(produced)} strategy(ies) in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
