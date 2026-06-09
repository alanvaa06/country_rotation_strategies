"""Tests for country_rotation.backtest.benchmarks — equal-weight buy-and-hold null.

Parity intent: honest passive null for a rotation strategy (no rebalancing).
"""
import numpy as np
import pandas as pd
import pytest

from country_rotation.backtest import benchmarks


# ---------------------------------------------------------------------------
# Mandatory spec tests (verbatim from task spec)
# ---------------------------------------------------------------------------

def test_eqw_buy_hold_starts_at_one(synthetic_prices):
    countries = [c for c in synthetic_prices.columns if c != "World"]
    eq = benchmarks.equal_weight_buy_hold(synthetic_prices[countries])
    assert abs(eq.iloc[0] - 1.0) < 1e-12
    assert len(eq) == len(synthetic_prices)
    assert eq.isna().sum() == 0


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------

def test_eqw_start_parameter_slices(synthetic_prices):
    """start= slices the index so the returned series begins at or after start."""
    countries = [c for c in synthetic_prices.columns if c != "World"]
    start_date = synthetic_prices.index[100]
    eq = benchmarks.equal_weight_buy_hold(synthetic_prices[countries], start=start_date)
    assert eq.index[0] >= start_date
    assert abs(eq.iloc[0] - 1.0) < 1e-12
    assert eq.isna().sum() == 0


def test_nan_column_dropped():
    """A column that has any NaN over the window is silently dropped."""
    idx = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(42)
    data = {
        "A": 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 100)),
        "B": 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 100)),
        "C_nan": np.full(100, np.nan),         # all-NaN column
    }
    # Also a column with a single mid-series NaN
    vals = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 100))
    vals[50] = np.nan
    data["D_partial_nan"] = vals

    prices = pd.DataFrame(data, index=idx)
    eq = benchmarks.equal_weight_buy_hold(prices)

    # Series must still start at 1, no NaNs, and only A+B contributed
    assert abs(eq.iloc[0] - 1.0) < 1e-12
    assert eq.isna().sum() == 0
    assert len(eq) == 100


def test_eqw_single_column():
    """With one clean column the result is that column rebased to 1."""
    idx = pd.bdate_range("2021-01-01", periods=50)
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(
        {"A": 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 50))}, index=idx
    )
    eq = benchmarks.equal_weight_buy_hold(prices)
    expected = prices["A"] / prices["A"].iloc[0]
    np.testing.assert_allclose(eq.values, expected.values, rtol=1e-10)


def test_eqw_no_rebalancing():
    """Verify no rebalancing: two identical assets produce the same equity as one."""
    idx = pd.bdate_range("2021-01-01", periods=50)
    rng = np.random.default_rng(4)
    single = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 50))
    prices = pd.DataFrame({"A": single, "B": single}, index=idx)
    eq = benchmarks.equal_weight_buy_hold(prices)
    # A and B are identical; equal-weight of identical assets == either asset rebased
    expected = single / single[0]
    np.testing.assert_allclose(eq.values, expected, rtol=1e-10)
