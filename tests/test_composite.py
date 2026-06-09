"""Tests for country_rotation.signals.composite — composite scoring pipeline.

TDD: these tests are written first and must FAIL before the implementation exists.

Parity notes (legacy FactorTransformer.py lines 422-613):
- weighted_metric_average: pd.concat -> multi-level columns -> multiply per metric ->
  groupby(level=1, axis=1).sum() collapses to (date x country) per factor.
- aggregate_by_category: simple mean across factors per category; unknown factors warned+skipped.
- composite_score: contributions = weight * score per category (absolute, not proportion);
  composite = sum of contributions across categories.
- normalize_cross_section: per-date min-max to [0,1]; all-equal date -> 0.5 (legacy line 582).
- rebase_contributions: scale each category's contribution by (norm_score / composite_score)
  so that rebased contributions sum to normalized score at each (date, country).

Divergences from task spec vs legacy:
- Legacy contribution_dict stored proportions (weight*score / total) per country. New API
  stores absolute weight*score per category (easier to decompose). Rebase formula adjusted
  accordingly: scale by norm/composite rather than multiply proportion by norm.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from country_rotation.signals import composite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _factor_results(idx, cols):
    rng = np.random.default_rng(4)
    mk = lambda: pd.DataFrame(rng.random((len(idx), len(cols))), index=idx, columns=cols)
    return {
        "PE": {m: mk() for m in ("zscore", "absolute_pct", "relative_rank", "delta_pct")},
        "ROE": {m: mk() for m in ("zscore", "absolute_pct", "relative_rank", "delta_pct")},
    }


# ---------------------------------------------------------------------------
# Mandated tests (from task spec)
# ---------------------------------------------------------------------------

def test_weighted_average_math():
    idx = pd.bdate_range("2020-01-01", periods=5)
    fr = _factor_results(idx, ["A", "B"])
    w = {"zscore": 0.5, "absolute_pct": 0.5, "relative_rank": 0.0, "delta_pct": 0.0}
    out = composite.weighted_metric_average(fr, w)
    expected = 0.5 * fr["PE"]["zscore"] + 0.5 * fr["PE"]["absolute_pct"]
    pd.testing.assert_frame_equal(out["PE"], expected)


def test_normalize_cross_section_bounds_and_ties():
    idx = pd.bdate_range("2020-01-01", periods=3)
    comp = pd.DataFrame({"A": [1.0, 2.0, 5.0], "B": [3.0, 2.0, 1.0], "C": [2.0, 2.0, 3.0]}, index=idx)
    norm = composite.normalize_cross_section(comp)
    assert norm.iloc[0].min() == 0.0 and norm.iloc[0].max() == 1.0
    assert (norm.iloc[1] == 0.5).all()  # all-equal date -> 0.5 (legacy behavior)


def test_composite_weights_applied():
    idx = pd.bdate_range("2020-01-01", periods=4)
    cat = {
        "Valuation": pd.DataFrame({"A": [1.0] * 4, "B": [0.0] * 4}, index=idx),
        "Momentum": pd.DataFrame({"A": [0.0] * 4, "B": [1.0] * 4}, index=idx),
    }
    comp, contrib = composite.composite_score(cat, {"Valuation": 0.7, "Momentum": 0.3})
    assert abs(comp["A"].iloc[0] - 0.7) < 1e-12
    assert abs(comp["B"].iloc[0] - 0.3) < 1e-12


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------

def test_aggregate_by_category_catalog_mapping():
    """PE -> Valuation, ROE -> Profitability via CATALOG; result has correct shape."""
    idx = pd.bdate_range("2020-01-01", periods=4)
    cols = ["A", "B", "C"]
    rng = np.random.default_rng(7)
    weighted = {
        "PE": pd.DataFrame(rng.random((4, 3)), index=idx, columns=cols),
        "ROE": pd.DataFrame(rng.random((4, 3)), index=idx, columns=cols),
    }
    result = composite.aggregate_by_category(weighted)
    # PE is Valuation, ROE is Profitability
    assert "Valuation" in result
    assert "Profitability" in result
    # Valuation score == PE score (only one factor)
    pd.testing.assert_frame_equal(result["Valuation"], weighted["PE"])
    # Profitability score == ROE score (only one factor)
    pd.testing.assert_frame_equal(result["Profitability"], weighted["ROE"])


def test_aggregate_by_category_mean_of_two():
    """When two factors map to the same category, result is their mean."""
    idx = pd.bdate_range("2020-01-01", periods=3)
    cols = ["A", "B"]
    df1 = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, index=idx)
    df2 = pd.DataFrame({"A": [3.0, 4.0, 5.0], "B": [6.0, 7.0, 8.0]}, index=idx)
    # PE and Fwd_PE both -> Valuation
    weighted = {"PE": df1, "Fwd_PE": df2}
    result = composite.aggregate_by_category(weighted)
    expected = (df1 + df2) / 2
    pd.testing.assert_frame_equal(result["Valuation"], expected)


def test_aggregate_by_category_unknown_factor_warns():
    """Unknown factor names (not in CATALOG) are skipped with a warning."""
    idx = pd.bdate_range("2020-01-01", periods=2)
    cols = ["A"]
    df_known = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
    df_unknown = pd.DataFrame({"A": [9.0, 9.0]}, index=idx)
    weighted = {"PE": df_known, "NOT_A_REAL_FACTOR": df_unknown}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = composite.aggregate_by_category(weighted)
    assert any("NOT_A_REAL_FACTOR" in str(warning.message) for warning in w)
    # Result only has Valuation (from PE); unknown factor not included
    assert "Valuation" in result
    # Valuation should only reflect the PE score (unknown factor dropped)
    pd.testing.assert_frame_equal(result["Valuation"], df_known)


def test_rebase_contributions_invariant():
    """Sum of rebased category contributions equals the normalized score at each (date, country)."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    rng = np.random.default_rng(42)
    cats = {
        "Valuation": pd.DataFrame(rng.random((5, 3)), index=idx, columns=["A", "B", "C"]),
        "Profitability": pd.DataFrame(rng.random((5, 3)), index=idx, columns=["A", "B", "C"]),
        "Momentum": pd.DataFrame(rng.random((5, 3)), index=idx, columns=["A", "B", "C"]),
    }
    weights = {"Valuation": 0.4, "Profitability": 0.3, "Momentum": 0.3}
    comp_df, contrib = composite.composite_score(cats, weights)
    norm_df = composite.normalize_cross_section(comp_df)
    rebased = composite.rebase_contributions(contrib, comp_df, norm_df)

    # Sum of rebased contributions across categories must equal normalized score
    rebased_sum = sum(rebased[cat] for cat in rebased)
    pd.testing.assert_frame_equal(
        rebased_sum,
        norm_df,
        check_names=False,
        atol=1e-12,
    )


def test_composite_score_contributions_sum_to_composite():
    """Contributions (weight * score) sum to the composite score at each (date, country)."""
    idx = pd.bdate_range("2020-01-01", periods=4)
    rng = np.random.default_rng(99)
    cats = {
        "Valuation": pd.DataFrame(rng.random((4, 2)), index=idx, columns=["X", "Y"]),
        "Momentum": pd.DataFrame(rng.random((4, 2)), index=idx, columns=["X", "Y"]),
    }
    weights = {"Valuation": 0.6, "Momentum": 0.4}
    comp_df, contrib = composite.composite_score(cats, weights)
    contrib_sum = sum(contrib[cat] for cat in contrib)
    pd.testing.assert_frame_equal(contrib_sum, comp_df, check_names=False, atol=1e-12)


def test_normalize_cross_section_values_in_01():
    """All normalized values should be in [0, 1]."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    rng = np.random.default_rng(11)
    comp = pd.DataFrame(rng.random((10, 5)), index=idx,
                        columns=["A", "B", "C", "D", "E"])
    norm = composite.normalize_cross_section(comp)
    assert (norm.values >= 0.0).all()
    assert (norm.values <= 1.0).all()


def test_weighted_average_missing_metric_weight_raises():
    """Metrics present in factor_results but absent from metric_weights raise ValueError."""
    idx = pd.bdate_range("2020-01-01", periods=3)
    fr = _factor_results(idx, ["A", "B"])
    incomplete = {"zscore": 0.5, "absolute_pct": 0.5}  # relative_rank, delta_pct missing
    with pytest.raises(ValueError, match="delta_pct.*relative_rank"):
        composite.weighted_metric_average(fr, incomplete)


def test_weighted_average_zero_weight_metric_excluded():
    """Metrics with weight=0 contribute nothing to the factor score."""
    idx = pd.bdate_range("2020-01-01", periods=3)
    cols = ["A", "B"]
    rng = np.random.default_rng(5)
    # Use fixed values so we can verify exactly
    df_zscore = pd.DataFrame({"A": [0.8, 0.7, 0.6], "B": [0.2, 0.3, 0.4]}, index=idx)
    df_other = pd.DataFrame({"A": [999.0, 999.0, 999.0], "B": [999.0, 999.0, 999.0]}, index=idx)
    fr = {
        "PE": {
            "zscore": df_zscore,
            "absolute_pct": df_other,
            "relative_rank": df_other,
            "delta_pct": df_other,
        }
    }
    w = {"zscore": 1.0, "absolute_pct": 0.0, "relative_rank": 0.0, "delta_pct": 0.0}
    out = composite.weighted_metric_average(fr, w)
    pd.testing.assert_frame_equal(out["PE"], df_zscore)
