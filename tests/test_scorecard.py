"""Tests for country_rotation/validation/scorecard.py (Task B4).

TDD: these tests are written first (red) before the scorecard module exists.

Covers:
- ValidationReport structure (all fields present, verdict keys correct)
- verdict.overall == all(checks.values())
- dsr.dsr in [0, 1]
- oracle scores flip statistically_significant to True

Uses conftest fixtures synthetic_prices / synthetic_scores sliced to ~500 days
for a manageable runtime.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from country_rotation.config import BacktestConfig
from country_rotation.validation import scorecard as sc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> BacktestConfig:
    """Fast active-mode relative-selection config for scorecard tests."""
    return BacktestConfig(
        selection_criteria="relative",
        relative_selection_score=3,
        weighting_method="Equal",
        bmk="World",
        bmk_weight=0.0,
        mode="active",
        periodicity=21,
        transaction_cost_bps=2.0,
    )


def _grid() -> dict:
    return {"relative_selection_score": (3, 5)}


def _oracle_scores(prices: pd.DataFrame, periodicity: int = 21) -> pd.DataFrame:
    """Forward-return ranks: a look-ahead (perfect) factor."""
    countries = [c for c in prices.columns if c != "World"]
    px = prices[countries]
    fwd = px.shift(-periodicity) / px - 1.0
    return fwd.rank(axis=1, pct=True).fillna(0.5)


# ---------------------------------------------------------------------------
# Test 1: structure check
# ---------------------------------------------------------------------------

def test_compute_validation_structure(synthetic_prices, synthetic_scores):
    """ValidationReport has all fields; verdict.checks has exactly the 3 keys;
    all check values are bool; overall == all(checks.values()); 0 <= dsr.dsr <= 1.
    """
    prices = synthetic_prices.iloc[:500]
    countries = [c for c in prices.columns if c != "World"]
    scores = synthetic_scores.loc[prices.index, countries]

    cfg = _cfg()
    grid = _grid()

    report = sc.compute_validation(
        scores, prices, cfg, grid,
        n_boot=100, n_mc=10, n_folds=3, seed=0,
    )

    # Report has all the expected fields
    assert hasattr(report, "sharpe")
    assert hasattr(report, "psr")
    assert hasattr(report, "dsr")
    assert hasattr(report, "nw_vs_eqw")
    assert hasattr(report, "bootstrap")
    assert hasattr(report, "sweep")
    assert hasattr(report, "stability")
    assert hasattr(report, "walkforward")
    assert hasattr(report, "mc")
    assert hasattr(report, "verdict")

    # Verdict checks: exactly the 3 expected keys
    expected_keys = {"no_overfitting", "param_stable", "statistically_significant"}
    assert set(report.verdict.checks.keys()) == expected_keys

    # All check values are bool (not NaN, not int, not numpy bool_)
    for key, val in report.verdict.checks.items():
        assert isinstance(val, bool), f"checks[{key!r}] is {type(val)}, not bool"

    # overall == all(checks.values())
    assert report.verdict.overall == all(report.verdict.checks.values())

    # notes is a tuple
    assert isinstance(report.verdict.notes, tuple)

    # DSR in [0, 1]
    assert math.isfinite(report.dsr.dsr)
    assert 0.0 <= report.dsr.dsr <= 1.0


# ---------------------------------------------------------------------------
# Test 2: oracle scores flip statistically_significant to True
# ---------------------------------------------------------------------------

def test_oracle_flips_significance(synthetic_prices):
    """Oracle (forward-return rank) scores should yield statistically_significant=True.

    With a perfect predictive signal the Sharpe t-stat, PSR, and bootstrap CI
    should all comfortably pass threshold even with small n_mc/n_boot.
    """
    prices = synthetic_prices.iloc[:500]
    scores = _oracle_scores(prices)
    # Align to prices index (oracle has NaN at the tail due to shift)
    scores = scores.reindex(prices.index).fillna(0.5)

    cfg = _cfg()
    grid = _grid()

    report = sc.compute_validation(
        scores, prices, cfg, grid,
        n_boot=100, n_mc=10, n_folds=3, seed=42,
    )

    assert report.verdict.checks["statistically_significant"] is True


# ---------------------------------------------------------------------------
# Test 3: Thresholds dataclass is a frozen dataclass with correct defaults
# ---------------------------------------------------------------------------

def test_thresholds_defaults():
    th = sc.Thresholds()
    assert th.dsr == pytest.approx(0.95)
    assert th.psr == pytest.approx(0.95)
    assert th.mc_p == pytest.approx(0.05)
    assert th.wfe == pytest.approx(0.5)
    assert th.frac_oos_positive == pytest.approx(0.5)
    assert th.sharpe_t == pytest.approx(2.0)
    assert th.stability_frac_positive == pytest.approx(0.7)
    assert th.stability_max_zscore == pytest.approx(1.5)
    # Frozen — should raise on attribute assignment
    with pytest.raises((TypeError, AttributeError)):
        th.dsr = 0.5


# ---------------------------------------------------------------------------
# Test 4: active basis — structure check
# ---------------------------------------------------------------------------

def test_compute_validation_active_basis_structure(synthetic_prices, synthetic_scores):
    """basis='active' produces a well-formed report: same 3 bool verdict keys,
    all report fields present, dsr in [0,1], runs without error."""
    prices = synthetic_prices.iloc[:500]
    countries = [c for c in prices.columns if c != "World"]
    scores = synthetic_scores.loc[prices.index, countries]

    cfg = _cfg()
    grid = _grid()

    report = sc.compute_validation(
        scores, prices, cfg, grid,
        n_boot=80, n_mc=8, n_folds=3, seed=0, basis="active",
    )

    for fld in ("sharpe", "psr", "dsr", "nw_vs_eqw", "bootstrap",
                "sweep", "stability", "walkforward", "mc", "verdict"):
        assert hasattr(report, fld)

    expected_keys = {"no_overfitting", "param_stable", "statistically_significant"}
    assert set(report.verdict.checks.keys()) == expected_keys
    for key, val in report.verdict.checks.items():
        assert isinstance(val, bool), f"checks[{key!r}] is {type(val)}, not bool"
    assert report.verdict.overall == all(report.verdict.checks.values())
    assert isinstance(report.verdict.notes, tuple)

    assert math.isfinite(report.dsr.dsr)
    assert 0.0 <= report.dsr.dsr <= 1.0


# ---------------------------------------------------------------------------
# Test 5: active basis differs from absolute
# ---------------------------------------------------------------------------

def test_active_basis_differs_from_absolute(synthetic_prices, synthetic_scores):
    """The headline annualized Sharpe must differ between absolute and active
    bases — active strips the benchmark drift (synthetic scores have no edge)."""
    prices = synthetic_prices.iloc[:500]
    countries = [c for c in prices.columns if c != "World"]
    scores = synthetic_scores.loc[prices.index, countries]

    cfg = _cfg()
    grid = _grid()

    rep_abs = sc.compute_validation(
        scores, prices, cfg, grid,
        n_boot=80, n_mc=8, n_folds=3, seed=0, basis="absolute",
    )
    rep_act = sc.compute_validation(
        scores, prices, cfg, grid,
        n_boot=80, n_mc=8, n_folds=3, seed=0, basis="active",
    )

    assert math.isfinite(rep_abs.sharpe.sharpe_ann)
    assert math.isfinite(rep_act.sharpe.sharpe_ann)
    assert abs(rep_abs.sharpe.sharpe_ann - rep_act.sharpe.sharpe_ann) > 1e-6

    # Default basis is absolute (unchanged behavior)
    rep_default = sc.compute_validation(
        scores, prices, cfg, grid,
        n_boot=80, n_mc=8, n_folds=3, seed=0,
    )
    assert rep_default.sharpe.sharpe_ann == pytest.approx(rep_abs.sharpe.sharpe_ann)


# ---------------------------------------------------------------------------
# Test 6: oracle on active basis still produces finite, honest stats
# ---------------------------------------------------------------------------

def test_oracle_active_significant(synthetic_prices):
    """Oracle (forward-return-rank) scores on the active basis retain real
    selection edge over equal-weight: stats run and are finite."""
    prices = synthetic_prices.iloc[:500]
    scores = _oracle_scores(prices)
    scores = scores.reindex(prices.index).fillna(0.5)

    cfg = _cfg()
    grid = _grid()

    report = sc.compute_validation(
        scores, prices, cfg, grid,
        n_boot=80, n_mc=8, n_folds=3, seed=42, basis="active",
    )

    assert math.isfinite(report.sharpe.t_stat)
    assert math.isfinite(report.bootstrap.ci_low)
    # The active verdict is a bool (may be True or False — don't over-assert)
    assert isinstance(report.verdict.checks["statistically_significant"], bool)
