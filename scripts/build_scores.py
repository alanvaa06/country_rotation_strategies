"""Thin CLI: run the full scoring pipeline and write NormalizedScores xlsx.

NOTE: This script requires local data folders (Inputs/, ProcessedInputs/) and
Classification.xlsx which are gitignored (data-only, not committed to the repo).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import pandas as pd

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_FILL_METRICS = (
    "EBITDA",
    "EV_EBITDA",
    "Fwd_EV_EBITDA",
    "Net_Debt_Ebitda",
    "PCF",
)

_DEFAULT_CONFIG = "configs/default.json"
_DEFAULT_MARKET = "World"
_DEFAULT_OUTPUT_DIR = "outputs/normalized_scores"


def _equal_weights(keys: list[str]) -> dict[str, float]:
    n = len(keys)
    return {k: 1.0 / n for k in keys} if n else {}


def _filter_countries(
    dfs: dict[str, pd.DataFrame],
    market: str,
    classification: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Select country columns that belong to *market* per the classification sheet.

    World → all countries in classification; DM/EM → filter by Segment column.
    """
    if market == "World":
        valid = set(classification.index.tolist())
    else:
        # Segment column distinguishes DM from EM; Region sub-divides EM
        if "Segment" in classification.columns:
            valid = set(classification.index[classification["Segment"] == market].tolist())
        elif "Region" in classification.columns:
            valid = set(classification.index[classification["Region"] == market].tolist())
        else:
            warnings.warn(
                f"Classification has no Segment/Region column; using all countries for {market!r}.",
                stacklevel=2,
            )
            valid = set(classification.index.tolist())

    out: dict[str, pd.DataFrame] = {}
    for key, df in dfs.items():
        cols = [c for c in df.columns if c in valid]
        out[key] = df[cols]
    return out


def main() -> None:  # noqa: C901 — sequential pipeline, complexity is expected
    parser = argparse.ArgumentParser(
        description="Build cross-sectional normalized scores from raw market data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Path to JSON platform config file.",
    )
    parser.add_argument(
        "--market",
        default=_DEFAULT_MARKET,
        help="Market segment to score: World | DM | EM.",
    )
    parser.add_argument(
        "--metric-weights",
        default=None,
        dest="metric_weights",
        help=(
            "JSON string mapping metric names to weights, e.g. "
            '\'{"zscore": 0.25, "absolute_pct": 0.25, '
            '"relative_rank": 0.25, "delta_pct": 0.25}\'. '
            "Defaults to equal weights across the 4 metrics."
        ),
    )
    parser.add_argument(
        "--category-weights",
        default=None,
        dest="category_weights",
        help=(
            "JSON string mapping category names to weights, e.g. "
            '\'{"Valuation": 0.25, "Quality": 0.25, "Profitability": 0.25, "Momentum": 0.25}\'. '
            "Defaults to equal weights across present categories."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        dest="output_dir",
        help="Directory where the output xlsx is written.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Lazy imports (keeps --help fast and import-side-effect free)
    # ------------------------------------------------------------------
    from country_rotation.config import load_config
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
    from country_rotation.factors.catalog import CATALOG, get_spec
    from country_rotation.factors.transforms import transform_factor
    from country_rotation.signals.composite import (
        aggregate_by_category,
        composite_score,
        normalize_cross_section,
        weighted_metric_average,
    )

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)

    # ------------------------------------------------------------------
    # Metric weights — keys match transform_factor output
    # ------------------------------------------------------------------
    _metrics = ["zscore", "absolute_pct", "relative_rank", "delta_pct"]
    if args.metric_weights is not None:
        metric_weights = json.loads(args.metric_weights)
    else:
        metric_weights = _equal_weights(_metrics)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    print(f"[build_scores] Reading inputs from '{cfg.data.inputs_folder}' …")
    raw = read_inputs(cfg.data.inputs_folder)
    classification = load_classification(cfg.data.classification_file)

    raw = remove_weekends(raw)
    raw = slice_by_date(raw, cfg.data.target_date)
    raw = drop_countries(raw, cfg.data.columns_to_drop)
    raw = apply_publication_lag(raw, cfg.data.publication_lag_days)
    raw = fill_gaps(raw, _FILL_METRICS, cfg.data.ffill_limit)

    # ------------------------------------------------------------------
    # Market filtering
    # ------------------------------------------------------------------
    print(f"[build_scores] Filtering for market='{args.market}' …")
    raw = _filter_countries(raw, args.market, classification)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    print("[build_scores] Running derived-metric processing …")
    processed = run_processing(raw, classification)

    # ------------------------------------------------------------------
    # Factor transforms — iterate over catalog factors present in processed
    # ------------------------------------------------------------------
    catalog_names = {spec.name for spec in CATALOG}
    present_factors = [f for f in processed if f in catalog_names]

    print(f"[build_scores] Transforming {len(present_factors)} catalog factors …")
    factor_results: dict[str, dict[str, pd.DataFrame]] = {}
    for factor_name in present_factors:
        spec = get_spec(factor_name)
        # transform_factor computes all 4 metrics and applies the direction flag
        factor_results[factor_name] = transform_factor(processed[factor_name], spec.direction)

    if not factor_results:
        print("[build_scores] WARNING: no catalog factors found in processed data. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Composite pipeline
    # ------------------------------------------------------------------
    print("[build_scores] Computing weighted metric average …")
    weighted = weighted_metric_average(factor_results, metric_weights)

    print("[build_scores] Aggregating by category …")
    category_scores = aggregate_by_category(weighted)

    # Category weights
    if args.category_weights is not None:
        cat_weights = json.loads(args.category_weights)
    else:
        cat_weights = _equal_weights(list(category_scores.keys()))

    print("[build_scores] Computing composite score …")
    composite, _contributions = composite_score(category_scores, cat_weights)

    print("[build_scores] Normalizing cross-section …")
    normalized = normalize_cross_section(composite)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"NormalizedScores_{args.market}_custom.xlsx")
    print(f"[build_scores] Writing output → {out_path}")
    normalized.to_excel(out_path)
    print("[build_scores] Done.")


if __name__ == "__main__":
    main()
