"""Per-segment research driver: screen -> compose -> validate -> report.

Runs the full evidence pipeline on the real (gitignored) ``Inputs/`` data for
one market segment (World / DM / EM):

1. Ingest + derived-metric processing (legacy-parity pipeline).
2. Universe extraction from ``Classification.xlsx`` (``Type == 'Country'``
   only — the sheet also contains pseudo-country Region rows: DM, EM, Europe,
   Asia, World, LatAm, which must NOT enter the cross-section).
3. Per-factor scores: 4 standardized metrics, equal-weighted (1/4 each).
4. OOS-honest walk-forward screening with a terminal lockbox
   (``selection.walkforward``); screening table written to xlsx.
5. Composite from KEPT factors only, balanced (equal) weights across the
   surviving categories — literature supports near-equal value/momentum
   weighting at the country level.
6. Full validation scorecard (``validation.scorecard.compute_validation``).
7. HTML report + machine-readable verdict JSON.

Benchmark construction
----------------------
The Engine needs a benchmark COLUMN in ``prices``.  We construct it as the
equal-weight average of universe price relatives — mean of
``px / px[first_valid]`` per country — named after the segment.  A
cap-weighted aggregate is not constructible from the inputs (no investable
country weights); the raw ``Price.xlsx`` index columns (``World``/``DM``/
``EM``) are external index levels that fall outside the classified universe
and are intentionally replaced so the benchmark matches the equal-weight
buy-and-hold null used inside the validation suite.

Usage
-----
    python scripts/research_run.py --segment World [--quick] [--periodicity 63]

Note: requires local data (``Inputs/``, ``Classification.xlsx``) which is
gitignored — present on the research machine only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import warnings

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DEFAULT_CONFIG = "configs/default.json"
_DEFAULT_OUTPUT_DIR = "outputs/research"

#: Metrics gap-filled at ingestion (parity with scripts/build_scores.py).
_FILL_METRICS = ("EBITDA", "EV_EBITDA", "Fwd_EV_EBITDA", "Net_Debt_Ebitda", "PCF")

#: Equal weights for the four standardized factor metrics.
_METRIC_WEIGHTS = {
    "zscore": 0.25,
    "absolute_pct": 0.25,
    "relative_rank": 0.25,
    "delta_pct": 0.25,
}

_MIN_COUNTRIES = 8
_MIN_FACTORS = 5

# Screening policy (Plan D step 1)
_SCREEN_N_FOLDS = 5
_SCREEN_LOCKBOX_FRAC = 0.2
_SCREEN_MIN_MEAN_IC = 0.03


def _log(msg: str) -> None:
    print(f"[research_run] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Universe / benchmark helpers
# ---------------------------------------------------------------------------

def extract_universe(classification, segment: str) -> list:
    """Country names for *segment* from the classification sheet.

    Only rows with ``Type == 'Country'`` qualify (the sheet also holds
    Region pseudo-rows).  World = all classified countries; DM / EM filter
    on the ``Segment`` column.
    """
    if "Type" in classification.columns:
        countries = classification[classification["Type"] == "Country"]
    else:
        countries = classification

    if segment == "World":
        return countries.index.tolist()
    if "Segment" not in countries.columns:
        raise ValueError("Classification sheet has no 'Segment' column.")
    return countries.index[countries["Segment"] == segment].tolist()


def equal_weight_index(prices):
    """Equal-weight average of price relatives (each country rebased to its
    first valid observation).  Used as the segment benchmark column."""
    import numpy as np

    def _base(col):
        first = col.first_valid_index()
        return col.loc[first] if first is not None else np.nan

    base = prices.apply(_base)
    return prices.div(base, axis=1).mean(axis=1)


def _json_safe(value):
    """Recursively convert numpy / NaN values into strict-JSON-safe types."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int,)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # numpy scalars
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def ingest(cfg):
    """Stage 1: read, clean and process all input metrics."""
    from country_rotation.data.ingestion import (
        apply_publication_lag,
        drop_countries,
        fill_gaps,
        load_classification,
        read_inputs,
        remove_weekends,
        slice_by_date,
    )
    from country_rotation.data.processing import run_processing

    _log(f"Reading inputs from '{cfg.data.inputs_folder}' ...")
    raw = read_inputs(cfg.data.inputs_folder)
    classification = load_classification(cfg.data.classification_file)

    raw = remove_weekends(raw)
    raw = slice_by_date(raw, cfg.data.target_date)
    raw = drop_countries(raw, cfg.data.columns_to_drop)
    raw = apply_publication_lag(raw, cfg.data.publication_lag_days)
    raw = fill_gaps(raw, _FILL_METRICS, cfg.data.ffill_limit)

    _log("Running derived-metric processing ...")
    processed = run_processing(raw, classification)
    return processed, classification


def build_factor_scores(processed, universe: list) -> tuple:
    """Stage 3: factor-level scores (equal-weighted 4-metric average).

    Returns (factor_scores dict, exploratory name set).  Factors whose
    transform fails or produces no data are skipped with a warning.
    """
    from country_rotation.factors.catalog import CATALOG
    from country_rotation.factors.transforms import transform_factor
    from country_rotation.signals.composite import weighted_metric_average

    factor_scores: dict = {}
    exploratory: set = set()

    for spec in CATALOG:
        if spec.name not in processed:
            continue
        df = processed[spec.name]
        cols = [c for c in universe if c in df.columns]
        if len(cols) < 3:
            warnings.warn(
                f"Factor '{spec.name}': only {len(cols)} universe columns. Skipping.",
                stacklevel=2,
            )
            continue
        try:
            metrics = transform_factor(df[cols], spec.direction)
            score = weighted_metric_average({spec.name: metrics}, _METRIC_WEIGHTS)[
                spec.name
            ]
        except Exception as exc:  # noqa: BLE001 — robustness per factor
            warnings.warn(
                f"Factor '{spec.name}': transform failed ({exc}). Skipping.",
                stacklevel=2,
            )
            continue
        if score.dropna(how="all").empty:
            warnings.warn(
                f"Factor '{spec.name}': all-NaN score. Skipping.", stacklevel=2
            )
            continue
        factor_scores[spec.name] = score
        if spec.exploratory:
            exploratory.add(spec.name)

    return factor_scores, exploratory


def run_screening(factor_scores: dict, exploratory: set, prices, periodicity: int):
    """Stage 4: walk-forward screening + one-shot lockbox verification.

    scipy ``ConstantInputWarning`` is silenced for this stage only: during
    the metric warm-up window all-NaN metrics aggregate to constant 0.0
    cross-sections (documented legacy behaviour of weighted_metric_average),
    which makes early-grid Spearman calls degenerate — those ICs are NaN and
    are dropped by the screen, so the warning is pure noise here.
    """
    from scipy.stats import ConstantInputWarning

    from country_rotation.selection.walkforward import (
        screen_factors,
        verify_on_lockbox,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = screen_factors(
            factor_scores,
            prices,
            periodicity=periodicity,
            n_folds=_SCREEN_N_FOLDS,
            lockbox_frac=_SCREEN_LOCKBOX_FRAC,
            min_mean_ic=_SCREEN_MIN_MEAN_IC,
            exploratory=exploratory,
        )
        result = verify_on_lockbox(
            result,
            factor_scores,
            prices,
            periodicity=periodicity,
            lockbox_frac=_SCREEN_LOCKBOX_FRAC,
        )
    return result


def screening_table(result):
    """Flatten a FactorScreenResult into one DataFrame for xlsx export."""
    import numpy as np

    table = result.fold_ic.copy()
    table["mean_ic"] = table.mean(axis=1, skipna=True)
    kept = set(result.kept)
    weak = set(result.weak)

    def _status(name: str) -> str:
        if name in weak:
            return "weak"
        if name in kept:
            return "kept"
        return "dropped"

    table["status"] = [_status(n) for n in table.index]
    table["bh_qvalue"] = [result.bh_qvalues.get(n, np.nan) for n in table.index]
    lockbox = result.lockbox_ic or {}
    table["lockbox_ic"] = [lockbox.get(n, np.nan) for n in table.index]
    return table.sort_values("mean_ic", ascending=False)


def build_composite(factor_scores: dict, kept: tuple) -> tuple:
    """Stage 5: composite from kept factors, balanced category weights.

    Returns (normalized scores, rebased contributions, category weights).
    """
    from country_rotation.signals.composite import (
        aggregate_by_category,
        composite_score,
        normalize_cross_section,
        rebase_contributions,
    )

    weighted = {name: factor_scores[name] for name in kept}
    category_scores = aggregate_by_category(weighted)

    # Balanced weights: equal weight per surviving category (near-equal
    # value/momentum weighting is the literature-supported default).
    n_cat = len(category_scores)
    category_weights = {cat: 1.0 / n_cat for cat in category_scores}
    _log(f"Surviving categories ({n_cat}): {sorted(category_scores)}")

    composite, contributions = composite_score(category_scores, category_weights)
    normalized = normalize_cross_section(composite)
    rebased = rebase_contributions(contributions, composite, normalized)
    return normalized, rebased, category_weights


# ---------------------------------------------------------------------------
# Verdict serialization
# ---------------------------------------------------------------------------

def build_verdict_payload(
    args,
    universe,
    screen_result,
    runtime_s,
    category_weights=None,
    validation_report=None,
) -> dict:
    """Assemble the verdict JSON payload.

    When ``validation_report`` is None (no factor survived screening) the
    verdict is an explicit negative: ``overall=False`` with an explanatory
    note and ``stats=None`` — the screening table is the evidence.
    """
    vr = validation_report
    if vr is not None:
        verdict = {
            "checks": vr.verdict.checks,
            "overall": vr.verdict.overall,
            "notes": list(vr.verdict.notes),
        }
        stats = {
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
    else:
        verdict = {
            "checks": {},
            "overall": False,
            "notes": [
                "No factors survived the walk-forward screen "
                "(mean-IC / sign-consistency / BH-FDR gates) — "
                "composite, validation and report stages skipped."
            ],
        }
        stats = None

    return _json_safe(
        {
            "segment": args.segment,
            "quick": bool(args.quick),
            "periodicity": args.periodicity,
            "n_countries": len(universe),
            "universe": list(universe),
            "screening": {
                "n_candidates": int(len(screen_result.fold_ic)),
                "min_mean_ic": _SCREEN_MIN_MEAN_IC,
                "kept": list(screen_result.kept),
                "weak": list(screen_result.weak),
                "dropped": list(screen_result.dropped),
                "bh_qvalues": screen_result.bh_qvalues,
                "lockbox_ic": screen_result.lockbox_ic,
            },
            "category_weights": category_weights,
            "verdict": verdict,
            "stats": stats,
            "runtime_seconds": round(runtime_s, 1),
        }
    )


def print_verdict_summary(payload: dict) -> None:
    v = payload["verdict"]
    s = payload["stats"]

    def _f(x, fmt="{:.3f}"):
        return fmt.format(x) if x is not None else "nan"

    _log("=" * 64)
    _log(f"VERDICT [{payload['segment']}] overall: "
         f"{'PASS' if v['overall'] else 'FAIL'}")
    for name, passed in v["checks"].items():
        _log(f"  check {name:28s}: {'PASS' if passed else 'FAIL'}")
    kept = payload["screening"]["kept"]
    weak = payload["screening"]["weak"]
    _log(f"  kept factors ({len(kept)}): {kept}")
    _log(f"  weak survivors ({len(weak)}): {weak}")
    if s is not None:
        _log(
            f"  sharpe_ann={_f(s['sharpe_ann'], '{:.2f}')} "
            f"t={_f(s['sharpe_t_stat'], '{:.2f}')} "
            f"psr={_f(s['psr'])} dsr={_f(s['dsr'])} "
            f"mc_p={_f(s['mc_p_value'])} wfe={_f(s['wf_efficiency'], '{:.2f}')}"
        )
        ci = s["bootstrap_ci"]
        _log(
            f"  bootstrap_ci=[{_f(ci[0], '{:.4f}')}, {_f(ci[1], '{:.4f}')}] "
            f"nw_t_vs_eqw={_f(s['nw_t_vs_eqw'], '{:.2f}')} "
            f"frac_oos_pos={_f(s['frac_oos_positive'], '{:.2f}')}"
        )
    for note in v["notes"]:
        _log(f"  note: {note}")
    _log("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901 — sequential research pipeline
    parser = argparse.ArgumentParser(
        description="Per-segment research pipeline: screen -> compose -> validate -> report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--segment", required=True, choices=("World", "DM", "EM"),
        help="Market segment universe.",
    )
    parser.add_argument(
        "--config", default=_DEFAULT_CONFIG, help="Path to JSON platform config."
    )
    parser.add_argument(
        "--output-dir", default=_DEFAULT_OUTPUT_DIR, dest="output_dir",
        help="Directory for screening xlsx, report html and verdict json.",
    )
    parser.add_argument(
        "--periodicity", type=int, default=63,
        help="Rebalance / IC grid step in trading days.",
    )
    parser.add_argument(
        "--quick", action="store_true", default=False,
        help="Smoke mode: smaller n_boot/n_mc/n_folds for the validation suite.",
    )
    args = parser.parse_args()

    t0 = time.time()

    from country_rotation.config import load_config
    from country_rotation.reporting.report import build_report
    from country_rotation.validation.scorecard import compute_validation

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Ingest + processing
    # ------------------------------------------------------------------
    processed, classification = ingest(cfg)

    # ------------------------------------------------------------------
    # 2. Universe + prices + benchmark column
    # ------------------------------------------------------------------
    if "Price" not in processed:
        raise SystemExit("[research_run] ERROR: 'Price' metric missing from inputs.")

    universe_all = extract_universe(classification, args.segment)
    price_df = processed["Price"]
    universe = [c for c in universe_all if c in price_df.columns]
    if len(universe) < _MIN_COUNTRIES:
        raise SystemExit(
            f"[research_run] ERROR: segment '{args.segment}' has only "
            f"{len(universe)} countries with prices (< {_MIN_COUNTRIES})."
        )
    _log(f"Universe '{args.segment}': {len(universe)} countries.")

    prices = price_df[universe].copy()
    prices_with_bmk = prices.copy()
    prices_with_bmk[args.segment] = equal_weight_index(prices)

    cfg_bt = dataclasses.replace(
        cfg.backtest,
        bmk=args.segment,
        bmk_weight=0.0,
        mode="active",
        periodicity=args.periodicity,
        selection_criteria="relative",
    )

    # ------------------------------------------------------------------
    # 3. Per-factor scores
    # ------------------------------------------------------------------
    _log("Transforming catalog factors ...")
    factor_scores, exploratory = build_factor_scores(processed, universe)
    _log(f"Factors with data: {len(factor_scores)} "
         f"({len(exploratory)} exploratory).")
    if len(factor_scores) < _MIN_FACTORS:
        raise SystemExit(
            f"[research_run] ERROR: only {len(factor_scores)} factors with data "
            f"(< {_MIN_FACTORS}). Check input files vs catalog names."
        )

    # ------------------------------------------------------------------
    # 4. Screening + lockbox
    # ------------------------------------------------------------------
    _log("Screening factors (anchored folds + lockbox) ...")
    screen_result = run_screening(
        factor_scores, exploratory, prices, args.periodicity
    )
    table = screening_table(screen_result)
    screen_path = os.path.join(args.output_dir, f"screening_{args.segment}.xlsx")
    table.to_excel(screen_path)
    _log(f"Screening table -> {screen_path}")
    _log(f"Kept {len(screen_result.kept)}/{len(factor_scores)} factors "
         f"(weak: {len(screen_result.weak)}).")

    verdict_path = os.path.join(args.output_dir, f"verdict_{args.segment}.json")

    if not screen_result.kept:
        # Honest negative outcome: record the evidence, skip downstream stages.
        _log(
            "WARNING: no factors survived screening — composite/validation/"
            f"report skipped. Evidence: {screen_path}"
        )
        payload = build_verdict_payload(
            args, universe, screen_result, time.time() - t0
        )
        with open(verdict_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        _log(f"Verdict -> {verdict_path}")
        print_verdict_summary(payload)
        _log(f"Done in {payload['runtime_seconds']}s.")
        return

    # ------------------------------------------------------------------
    # 5. Composite from kept factors
    # ------------------------------------------------------------------
    _log("Building composite from kept factors ...")
    normalized, contributions, category_weights = build_composite(
        factor_scores, screen_result.kept
    )

    # ------------------------------------------------------------------
    # 6. Validation
    # ------------------------------------------------------------------
    n_boot = 50 if args.quick else 500
    n_mc = 10 if args.quick else 100
    n_folds = 3 if args.quick else 5
    grid = {"relative_selection_score": (3, 5, 7), "periodicity": (21, 63)}
    _log(
        f"Running validation suite (n_boot={n_boot}, n_mc={n_mc}, "
        f"n_folds={n_folds}) — this is the slow part ..."
    )
    validation_report = compute_validation(
        normalized,
        prices_with_bmk,
        cfg_bt,
        grid,
        n_boot=n_boot,
        n_mc=n_mc,
        n_folds=n_folds,
        seed=42,
    )

    # ------------------------------------------------------------------
    # 7. Report + verdict JSON
    # ------------------------------------------------------------------
    report_path = os.path.join(args.output_dir, f"report_{args.segment}.html")
    _log("Building HTML report ...")
    report_info = build_report(
        normalized,
        prices_with_bmk,
        cfg_bt,
        report_path,
        validation_report=validation_report,
        contributions=contributions,
    )
    _log(f"Report -> {report_info['path']} ({report_info['n_figures']} figures)")

    runtime_s = time.time() - t0
    payload = build_verdict_payload(
        args,
        universe,
        screen_result,
        runtime_s,
        category_weights=category_weights,
        validation_report=validation_report,
    )
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _log(f"Verdict -> {verdict_path}")

    print_verdict_summary(payload)
    _log(f"Done in {runtime_s:.1f}s.")


if __name__ == "__main__":
    main()
