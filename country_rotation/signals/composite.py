"""Composite scoring pipeline — pure functions ported from legacy FactorTransformer.py.

Porting source: FactorTransformer.py lines 422–613.

Functions
---------
weighted_metric_average   lines 422-460  (weighted sum of metric DataFrames per factor)
aggregate_by_category     lines 462-515  (mean across factors per category; catalog-driven)
composite_score           lines 517-551  (category-weight linear combination + contributions)
normalize_cross_section   lines 570-582  (per-date min-max → [0,1]; all-equal → 0.5)
rebase_contributions      lines 583-613  (scale contributions so they sum to normalized score)

Divergences from legacy
-----------------------
- aggregate_by_category now derives the factor→category map from ``CATALOG`` (a tuple of
  FactorSpec) rather than hardcoded lists.  Unknown factor names trigger warnings.warn instead
  of print().
- composite_score stores contributions as {category: DataFrame(dates×countries)} with values
  equal to ``weight × score`` (absolute, not proportion).  Legacy stored proportions
  (weight×score / total) per country; the new layout is more natural for decomposition.
  The rebase invariant is preserved: after rebase_contributions the sum of rebased category
  contributions at every (date, country) equals the normalized score.
- normalize_cross_section and rebase_contributions are separated into two pure functions
  so callers can pipeline them independently.  Legacy combined both in one method.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from country_rotation.factors.catalog import CATALOG, FactorSpec


# ---------------------------------------------------------------------------
# Step 1 — weighted metric average (per factor)
# ---------------------------------------------------------------------------

def weighted_metric_average(
    factor_results: dict[str, dict[str, pd.DataFrame]],
    metric_weights: dict[str, float],
) -> dict[str, pd.DataFrame]:
    """Compute a weighted average across metrics for each factor.

    For every factor the function forms a multi-level DataFrame
    ``(metric, country)`` by concatenating the per-metric DataFrames, then
    multiplies each metric slice by its weight and collapses to ``(date × country)``
    by summing across the metric level.

    Parity: FactorTransformer.calculate_weighted_average, lines 422-460.
    - pd.concat(metrics_dict, axis=1)                → MultiIndex(metric, country)
    - weighted_df[metric] *= weight                   → in-place scale per metric
    - groupby(level=1, axis=1).sum()                  → sum across metrics per country

    NaN handling: pandas ``groupby.sum()`` drops NaN by default (``min_count=0``),
    so a NaN metric for a country on a date contributes 0.  This matches legacy
    behaviour where non-zero weights for present metrics dominate.

    Strictness: every metric key present in *factor_results* must have an
    explicit entry in *metric_weights*.  A missing entry would otherwise be
    summed unscaled (implicit weight 1.0), silently distorting the factor
    score, so a ``ValueError`` is raised listing the offending metrics.

    Args:
        factor_results: Nested dict ``{factor_name: {metric_name: DataFrame(date×country)}}``.
        metric_weights: ``{metric_name: float}`` weights covering every metric
            present in *factor_results*; need not sum to 1.0 (warning if not).

    Returns:
        ``{factor_name: DataFrame(date×country)}`` — one weighted-score frame per factor.

    Raises:
        ValueError: If any metric key in *factor_results* is absent from
            *metric_weights*.
    """
    present_metrics: set[str] = {
        metric for metrics_dict in factor_results.values() for metric in metrics_dict
    }
    unweighted = sorted(present_metrics - set(metric_weights))
    if unweighted:
        raise ValueError(
            "metric_weights is missing an explicit weight for metric(s) present "
            f"in factor_results: {unweighted}. Unweighted metrics would be summed "
            "with an implicit weight of 1.0."
        )

    weight_sum = sum(metric_weights.values())
    if not np.isclose(weight_sum, 1.0):
        warnings.warn(
            f"metric_weights do not sum to 1.0 (sum={weight_sum:.6f}).",
            stacklevel=2,
        )

    final_scores: dict[str, pd.DataFrame] = {}
    for factor_name, metrics_dict in factor_results.items():
        try:
            all_metrics_df = pd.concat(metrics_dict, axis=1)
        except pd.errors.InvalidIndexError:
            warnings.warn(
                f"Could not concatenate metric DataFrames for factor '{factor_name}'. "
                f"Available metrics: {list(metrics_dict.keys())}. Skipping.",
                stacklevel=2,
            )
            continue

        weighted_df = all_metrics_df.copy()
        for metric_name, weight in metric_weights.items():
            if metric_name in weighted_df.columns.get_level_values(0):
                weighted_df[metric_name] = weighted_df[metric_name] * weight
            else:
                warnings.warn(
                    f"Metric '{metric_name}' listed in metric_weights not found for "
                    f"factor '{factor_name}'.",
                    stacklevel=2,
                )

        # Sum across metrics (level 0) to get one column per country (level 1).
        # Transpose-groupby-transpose avoids the deprecated groupby(axis=1).
        final_factor_score = weighted_df.T.groupby(level=1).sum().T
        final_scores[factor_name] = final_factor_score

    return final_scores


# ---------------------------------------------------------------------------
# Step 2 — aggregate by category
# ---------------------------------------------------------------------------

def aggregate_by_category(
    weighted: dict[str, pd.DataFrame],
    catalog: Sequence[FactorSpec] = CATALOG,
) -> dict[str, pd.DataFrame]:
    """Aggregate factor scores into category-level scores by taking the mean.

    The factor→category mapping is derived from ``catalog`` (a sequence of
    :class:`~country_rotation.factors.catalog.FactorSpec`).  Factors whose
    names do not appear in the catalog are skipped with a ``warnings.warn``.

    Parity: FactorTransformer.aggregate_by_category, lines 462-515.
    - factor_map.get(factor_name) → catalog lookup
    - mean per category = sum(dfs) / len(dfs)  (legacy lines 490-493)

    Args:
        weighted: ``{factor_name: DataFrame(date×country)}`` from
            :func:`weighted_metric_average`.
        catalog: Sequence of :class:`FactorSpec` used to resolve categories.
            Defaults to the global ``CATALOG``.

    Returns:
        ``{category_name: DataFrame(date×country)}`` — one aggregated score
        frame per category.
    """
    factor_map: dict[str, str] = {spec.name: spec.category for spec in catalog}

    # Group DataFrames by category
    category_frames: dict[str, list[pd.DataFrame]] = {}
    for factor_name, score_df in weighted.items():
        category = factor_map.get(factor_name)
        if category is None:
            warnings.warn(
                f"Factor '{factor_name}' not found in catalog. Skipping.",
                stacklevel=2,
            )
            continue
        category_frames.setdefault(category, []).append(score_df)

    # Mean across factors within each category (legacy: sum then divide by count)
    category_scores: dict[str, pd.DataFrame] = {}
    for category, dfs in category_frames.items():
        aligned_sum = dfs[0].copy() * 0
        for df in dfs:
            aligned_sum = aligned_sum.add(df, fill_value=0)
        category_scores[category] = aligned_sum / len(dfs)

    return category_scores


# ---------------------------------------------------------------------------
# Step 3 — composite score
# ---------------------------------------------------------------------------

def composite_score(
    category_scores: dict[str, pd.DataFrame],
    category_weights: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Combine category scores into a single composite score with contributions.

    composite[date, country] = Σ_c  category_weights[c] * category_scores[c][date, country]

    Contributions are stored as absolute values:
        contributions[c][date, country] = category_weights[c] * category_scores[c][date, country]

    So ``Σ_c contributions[c] == composite`` exactly (unlike legacy which stored
    proportions that sum to 1.0 per country-date).

    Parity: FactorTransformer.calculate_composite_score, lines 517-551.
    The composite arithmetic is identical; the contribution representation is
    reformulated from per-country-proportion to per-category-absolute to support
    the new API signatures (category DataFrames with countries as columns).

    Args:
        category_scores: ``{category: DataFrame(date×country)}``.
        category_weights: ``{category: float}`` weights.

    Returns:
        Tuple of:
        - ``composite`` — DataFrame(date×country) of composite scores.
        - ``contributions`` — ``{category: DataFrame(date×country)}`` where each
          cell = ``weight × score`` so contributions sum to composite across categories.
    """
    weight_sum = sum(category_weights.values())
    if not np.isclose(weight_sum, 1.0):
        warnings.warn(
            f"category_weights do not sum to 1.0 (sum={weight_sum:.6f}).",
            stacklevel=2,
        )

    contributions: dict[str, pd.DataFrame] = {}
    composite_parts: list[pd.DataFrame] = []

    for category, score_df in category_scores.items():
        weight = category_weights.get(category, 0.0)
        contrib = score_df * weight
        contributions[category] = contrib
        composite_parts.append(contrib)

    # Sum contributions across categories (align on index/columns)
    composite = composite_parts[0].copy()
    for part in composite_parts[1:]:
        composite = composite.add(part, fill_value=0.0)

    return composite, contributions


# ---------------------------------------------------------------------------
# Step 4 — cross-sectional normalization
# ---------------------------------------------------------------------------

def normalize_cross_section(composite: pd.DataFrame) -> pd.DataFrame:
    """Normalize composite scores cross-sectionally per date using min-max scaling.

    For each date:
    - If max != min: normalized = (x - min) / (max - min)  → [0, 1]
    - If max == min (all countries equal): normalized = 0.5  (legacy line 582)

    Parity: FactorTransformer.normalize_and_rebase_contributions, lines 570-582.

    Args:
        composite: DataFrame(date×country) of composite scores.

    Returns:
        DataFrame of the same shape with values in [0, 1].
    """
    normalized = composite.copy()
    for date in normalized.index:
        date_values = normalized.loc[date]
        min_val = date_values.min()
        max_val = date_values.max()
        if max_val != min_val:
            normalized.loc[date] = (date_values - min_val) / (max_val - min_val)
        else:
            normalized.loc[date] = 0.5
    return normalized


# ---------------------------------------------------------------------------
# Step 5 — rebase contributions
# ---------------------------------------------------------------------------

def rebase_contributions(
    contributions: dict[str, pd.DataFrame],
    composite: pd.DataFrame,
    normalized: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Scale category contributions so they sum to the normalized score.

    For each (date, country):
        rebased[c] = contributions[c] * (normalized / composite)

    When composite == 0 at a cell the scale factor is undefined; those cells
    become NaN (preserve undefined rather than produce Inf).

    Parity: FactorTransformer.normalize_and_rebase_contributions, lines 583-613.
    Legacy formula: ``rebased = original_contributions * norm_score`` where
    original_contributions were proportions summing to 1.0.  Equivalent here
    because proportions = contributions / composite, so:
        proportion * norm = (contrib / composite) * norm = contrib * (norm / composite)

    Invariant: Σ_c rebased[c] == normalized at every (date, country) where
    composite != 0 and no NaN is present.

    Args:
        contributions: ``{category: DataFrame(date×country)}`` from
            :func:`composite_score`.
        composite: DataFrame(date×country) of composite scores (denominator).
        normalized: DataFrame(date×country) of normalized scores (numerator target).

    Returns:
        ``{category: DataFrame(date×country)}`` of rebased contributions.
    """
    # scale factor: normalized / composite, NaN where composite == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = normalized.div(composite.replace(0, np.nan))

    rebased: dict[str, pd.DataFrame] = {}
    for category, contrib_df in contributions.items():
        rebased[category] = contrib_df.mul(scale)

    return rebased
