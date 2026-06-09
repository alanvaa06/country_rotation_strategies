"""Tests for country_rotation.data.processing — pure derived-metric functions.

TDD: these tests are written first and must FAIL before the implementation exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from country_rotation.data import processing


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _dfs():
    idx = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(3)
    px = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300)),
                       "B": 100 * np.cumprod(1 + rng.normal(0.0, 0.01, 300))}, index=idx)
    pe = pd.DataFrame({"A": np.full(300, 15.0), "B": np.full(300, -5.0)}, index=idx)
    return {"Price": px, "PE": pe}


# ---------------------------------------------------------------------------
# Provided tests (verbatim from spec)
# ---------------------------------------------------------------------------

def test_earnings_yield_safe_division():
    out = processing.add_yields(_dfs())
    ey = out["EarningsYieldTTM"]
    assert abs(ey["A"].iloc[0] - 1 / 15.0) < 1e-12
    assert np.isnan(ey["B"].iloc[0])  # negative PE -> NaN, not -0.2


def test_momentum_12_1_window_and_skip():
    dfs = _dfs()
    out = processing.add_price_momentum(dfs)
    mom = out["Momentum_12_1"]
    px = dfs["Price"]["A"]
    t = 280
    expected = px.iloc[t - 21] / px.iloc[t - 252] - 1.0
    assert abs(mom["A"].iloc[t] - expected) < 1e-12
    assert np.isnan(mom["A"].iloc[100])  # warmup


# ---------------------------------------------------------------------------
# Additional tests (4 focused tests added by implementor)
# ---------------------------------------------------------------------------

def test_consensus_growth_formula():
    """ConsensusSalesGrowth = PS/Fwd_PS - 1 (legacy formula)."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    ps = pd.DataFrame({"X": np.full(10, 20.0)}, index=idx)
    fwd_ps = pd.DataFrame({"X": np.full(10, 16.0)}, index=idx)
    dfs = {"PS": ps, "Fwd_PS": fwd_ps}
    out = processing.add_consensus_growth(dfs)
    # 20/16 - 1 = 0.25
    expected = 20.0 / 16.0 - 1.0
    result = out["ConsensusSalesGrowth"]["X"].iloc[0]
    assert abs(result - expected) < 1e-12


def test_spread_with_ten_year_as_dataframe():
    """EarningsYieldTTMSpread = EarningsYieldTTM - Ten_Year (fill_value=0 on sub)."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    pe = pd.DataFrame({"X": np.full(10, 10.0), "Y": np.full(10, 20.0)}, index=idx)
    # Ten_Year only has column X (no Y) — fill_value=0 means Y spread = EY(Y) - 0
    ten_yr = pd.DataFrame({"X": np.full(10, 0.03)}, index=idx)
    dfs = {"PE": pe, "Ten_Year": ten_yr}
    dfs = processing.add_yields(dfs)    # creates EarningsYieldTTM
    out = processing.add_spreads(dfs)

    spread_x = out["EarningsYieldTTMSpread"]["X"].iloc[0]
    # EY for X = 1/10 = 0.1; spread = 0.1 - 0.03 = 0.07
    assert abs(spread_x - 0.07) < 1e-12

    spread_y = out["EarningsYieldTTMSpread"]["Y"].iloc[0]
    # Ten_Year has no Y column -> fill_value=0, so spread = EY(Y) - 0 = 0.05
    assert abs(spread_y - 1.0 / 20.0) < 1e-12


def test_rolling_vol_formula():
    """RollingVol = Price.pct_change().rolling(63).std() * sqrt(252)."""
    idx = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(42)
    px = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))}, index=idx)
    dfs = {"Price": px}

    # We need Earnings too if processing requires it — provide a minimal valid set
    # add_rolling_stats only needs Price (and optionally Earnings/FwdEarnings/Flows)
    out = processing.add_rolling_stats(dfs)

    rv = out["RollingVol"]
    returns = px.pct_change()
    expected_vol = returns.rolling(63).std() * np.sqrt(252)

    # Check at a point well past warmup
    t = 100
    assert abs(rv["A"].iloc[t] - expected_vol["A"].iloc[t]) < 1e-12
    # Warmup rows should be NaN
    assert np.isnan(rv["A"].iloc[0])


def test_run_processing_minimal_inputs_and_momentum_keys():
    """run_processing should not crash with only Price+PE, and produce momentum keys."""
    idx = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(7)
    px = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300))}, index=idx)
    pe = pd.DataFrame({"A": np.full(300, 15.0)}, index=idx)
    dfs = {"Price": px, "PE": pe}

    # Minimal classification: just needs to exist (can be empty)
    classification = pd.DataFrame(
        {"Segment": ["DM"], "Region": ["Europe"]},
        index=["A"],
    )

    out = processing.run_processing(dfs, classification)

    # Momentum keys must be present
    assert "Momentum_12_1" in out
    assert "Momentum_6_1" in out
    # EarningsYieldTTM should also be derived
    assert "EarningsYieldTTM" in out
    # No crash is itself a pass
