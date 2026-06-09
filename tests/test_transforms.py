"""Tests for country_rotation.factors.transforms — pure transform functions.

TDD: these tests are written first and must FAIL before the implementation exists.

Parity notes (legacy FactorTransformer.py lines 314-366):
- zscore: expanding(min_periods=63).mean/std → shift(1) → norm.cdf
  At row t, stats are computed over rows 0..t-1 (expanding through t-1, shifted by 1).
  Test formula: hist = col.iloc[:t] (exclusive of t) matches legacy exactly.
- absolute_pct: expanding().rank(pct=True, method='min')
  method='min' => fraction of values strictly <= current value (rank of current in history).
  Test formula: (col.iloc[:t+1] <= col.iloc[t]).mean() matches method='min' pct rank.
- relative_rank: rank(axis=1, pct=True, method='average') — cross-sectional.
  Test checks that the country with the highest raw value has the highest rank.
- delta_pct: pct_change(63) then expanding().rank(pct=True, method='average').
  No-lookahead test uses same perturbation pattern as zscore.
- direction=-1: applies 1.0 - transform to all four outputs.

Pure functions do NOT apply .iloc[window:] slicing — they return full aligned DataFrames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from country_rotation.factors import transforms


# ---------------------------------------------------------------------------
# zscore_percentile
# ---------------------------------------------------------------------------

def test_zscore_uses_only_past_stats(synthetic_factor):
    """At row t, zscore uses mean/std of rows 0..t-1 (legacy shift(1) parity)."""
    z = transforms.zscore_percentile(synthetic_factor)
    col = synthetic_factor["A"]
    t = 150
    # hist = rows 0..t-1, exactly what expanding().shift(1) gives at index t
    hist = col.iloc[:t]
    expected = norm.cdf((col.iloc[t] - hist.mean()) / hist.std())
    assert abs(z["A"].iloc[t] - expected) < 1e-9


def test_zscore_no_lookahead(synthetic_factor):
    """Perturbing future rows must not change past outputs."""
    z1 = transforms.zscore_percentile(synthetic_factor)
    pert = synthetic_factor.copy()
    pert.iloc[200:] += 1000.0
    z2 = transforms.zscore_percentile(pert)
    pd.testing.assert_frame_equal(z1.iloc[:200], z2.iloc[:200])


def test_zscore_nan_before_min_periods(synthetic_factor):
    """Rows with fewer than min_periods=63 prior observations must be NaN."""
    z = transforms.zscore_percentile(synthetic_factor)
    # Row 0 has 0 prior rows -> NaN; row 62 has 62 prior rows -> still NaN
    # (shift(1) means at row 63 we have exactly 63 history rows [0..62])
    assert z["A"].iloc[0:63].isna().all()


# ---------------------------------------------------------------------------
# absolute_percentile
# ---------------------------------------------------------------------------

def test_absolute_percentile_expanding(synthetic_factor):
    """
    Parity with expanding().rank(pct=True, method='min'):
    at row t, value = fraction of col.iloc[0..t] that are <= col.iloc[t].
    """
    p = transforms.absolute_percentile(synthetic_factor)
    col = synthetic_factor["A"]
    t = 100
    # method='min': fraction of values (inclusive of self) that are <= current
    expected = (col.iloc[: t + 1] <= col.iloc[t]).mean()
    assert abs(p["A"].iloc[t] - expected) < 1e-9


def test_absolute_percentile_no_lookahead(synthetic_factor):
    """Perturbing future rows must not change past outputs."""
    p1 = transforms.absolute_percentile(synthetic_factor)
    pert = synthetic_factor.copy()
    pert.iloc[200:] += 1000.0
    p2 = transforms.absolute_percentile(pert)
    pd.testing.assert_frame_equal(p1.iloc[:200], p2.iloc[:200])


# ---------------------------------------------------------------------------
# relative_rank
# ---------------------------------------------------------------------------

def test_relative_rank_cross_sectional(synthetic_factor):
    """Country with the highest raw value at a given row has the highest rank."""
    r = transforms.relative_rank(synthetic_factor)
    row = synthetic_factor.iloc[250]
    # Drop NaN columns for the comparison (col F has NaN only in first 40 rows,
    # row 250 is fine — but be defensive)
    valid_row = row.dropna()
    valid_r = r.iloc[250][valid_row.index]
    assert valid_r[valid_row.idxmax()] == valid_r.max()


def test_relative_rank_bounded(synthetic_factor):
    """All relative rank values must be in (0, 1] (pct=True)."""
    r = transforms.relative_rank(synthetic_factor)
    vals = r.values.ravel()
    vals = vals[~np.isnan(vals)]
    assert (vals > 0).all() and (vals <= 1.0).all()


# ---------------------------------------------------------------------------
# delta_percentile
# ---------------------------------------------------------------------------

def test_delta_percentile_no_lookahead(synthetic_factor):
    """Perturbing future rows must not change past outputs."""
    d1 = transforms.delta_percentile(synthetic_factor)
    pert = synthetic_factor.copy()
    pert.iloc[200:] += 1000.0
    d2 = transforms.delta_percentile(pert)
    pd.testing.assert_frame_equal(d1.iloc[:200], d2.iloc[:200])


def test_delta_percentile_nan_before_window(synthetic_factor):
    """First window=63 rows of delta must be NaN (pct_change(63) has no prior)."""
    d = transforms.delta_percentile(synthetic_factor)
    # pct_change(63): first 63 rows are NaN, so expanding rank has nothing to rank
    assert d["A"].iloc[0:63].isna().all()


# ---------------------------------------------------------------------------
# apply_direction
# ---------------------------------------------------------------------------

def test_direction_inversion(synthetic_factor):
    """direction=-1 must flip all four transforms: value -> 1.0 - value."""
    base = transforms.transform_factor(synthetic_factor, direction=1)
    inv = transforms.transform_factor(synthetic_factor, direction=-1)
    pd.testing.assert_frame_equal(inv["zscore"], 1.0 - base["zscore"])
    pd.testing.assert_frame_equal(inv["absolute_pct"], 1.0 - base["absolute_pct"])
    pd.testing.assert_frame_equal(inv["relative_rank"], 1.0 - base["relative_rank"])
    pd.testing.assert_frame_equal(inv["delta_pct"], 1.0 - base["delta_pct"])


def test_direction_no_inversion(synthetic_factor):
    """direction=1 must return transforms unchanged."""
    base = transforms.transform_factor(synthetic_factor, direction=1)
    direct = transforms.apply_direction(base["zscore"], direction=1)
    pd.testing.assert_frame_equal(direct, base["zscore"])


# ---------------------------------------------------------------------------
# transform_factor — output shape and keys
# ---------------------------------------------------------------------------

def test_transform_factor_keys(synthetic_factor):
    """transform_factor must return dict with exactly the four expected keys."""
    result = transforms.transform_factor(synthetic_factor, direction=1)
    assert set(result.keys()) == {"zscore", "absolute_pct", "relative_rank", "delta_pct"}


def test_transform_factor_index_preserved(synthetic_factor):
    """All output DataFrames must share the same index as the input."""
    result = transforms.transform_factor(synthetic_factor, direction=1)
    for key, df in result.items():
        pd.testing.assert_index_equal(df.index, synthetic_factor.index,
                                      obj=f"transform_factor['{key}'].index")


def test_transform_factor_columns_preserved(synthetic_factor):
    """All output DataFrames must have the same columns as the input."""
    result = transforms.transform_factor(synthetic_factor, direction=1)
    for key, df in result.items():
        pd.testing.assert_index_equal(df.columns, synthetic_factor.columns,
                                      obj=f"transform_factor['{key}'].columns")
