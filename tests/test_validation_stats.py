"""Tests for country_rotation/validation/statistics.py (Task B1).

Five tests — one per statistic — written TDD-first.
All use the _equity_from_daily helper to convert daily returns to an equity curve.
"""
import math

import numpy as np
import pandas as pd
import pytest

from country_rotation.validation import statistics as val


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _equity_from_daily(daily_returns: np.ndarray) -> pd.Series:
    """Convert an array of daily simple returns to a pd.Series equity curve.

    The curve starts at 1.0 (index position 0) and compounds forward.
    Returns are indexed by integers starting at 0.
    """
    n = len(daily_returns)
    equity = np.empty(n + 1)
    equity[0] = 1.0
    for i, r in enumerate(daily_returns):
        equity[i + 1] = equity[i] * (1.0 + r)
    return pd.Series(equity, index=range(n + 1))


# ---------------------------------------------------------------------------
# Test 1 — sharpe_significance: t-stat scales with sqrt(n)
# ---------------------------------------------------------------------------

def test_sharpe_tstat_scales_with_sqrt_n():
    """Lo (2002) SE = sqrt((1 + SR^2/2) / n).

    For IID returns the t-stat scales with sqrt(n).  We verify this by
    constructing a 4× series as four copies of the same small series so
    that SR_small ≈ SR_large and the t-stat ratio is close to sqrt(4) = 2.
    """
    rng = np.random.default_rng(42)
    mu, sigma = 0.0005, 0.01   # positive drift, realistic daily vol
    r_small = rng.normal(mu, sigma, 252)
    # Tile to get exactly 4× the observations with the same mean/std
    r_large = np.tile(r_small, 4)

    eq_small = _equity_from_daily(r_small)
    eq_large = _equity_from_daily(r_large)

    res_small = val.sharpe_significance(eq_small)
    res_large = val.sharpe_significance(eq_large)

    # t should be larger for the 4× series
    assert res_large.t_stat > res_small.t_stat

    # Because SR is identical, ratio ≈ sqrt(4) = 2; allow 10% tolerance
    ratio = res_large.t_stat / res_small.t_stat
    assert 1.5 < ratio < 2.5, f"t-stat ratio {ratio:.3f} not near sqrt(4)=2.0"

    # SE check: Lo formula SE ~ sqrt((1 + SR^2/2) / n)
    n_small = len(r_small)
    sr_d = res_small.sharpe_daily
    expected_se = math.sqrt((1 + sr_d ** 2 / 2) / n_small)
    assert abs(res_small.se - expected_se) < 1e-10, (
        f"SE {res_small.se} != Lo formula {expected_se}"
    )

    # NaN-safety: flat equity (all zero returns)
    flat = _equity_from_daily(np.zeros(100))
    res_flat = val.sharpe_significance(flat)
    assert math.isnan(res_flat.t_stat) or res_flat.t_stat == 0.0

    # NaN-safety: single-period equity
    single = pd.Series([1.0, 1.01])
    res_single = val.sharpe_significance(single)
    assert math.isnan(res_single.t_stat) or res_single.t_stat == 0.0


# ---------------------------------------------------------------------------
# Test 2 — probabilistic_sharpe_ratio: monotonic and bounded in (0, 1)
# ---------------------------------------------------------------------------

def test_psr_monotonic_and_bounded():
    """PSR(SR* | SR_benchmark) is bounded in (0,1) and increases with observed SR."""
    rng = np.random.default_rng(7)
    n = 500
    benchmark_sr = 0.0  # daily

    # Three equity curves with increasing realized Sharpes
    low_eq  = _equity_from_daily(rng.normal(-0.0002, 0.01, n))
    mid_eq  = _equity_from_daily(rng.normal( 0.0005, 0.01, n))
    high_eq = _equity_from_daily(rng.normal( 0.0010, 0.01, n))

    psr_low  = val.probabilistic_sharpe_ratio(low_eq,  benchmark_sharpe=benchmark_sr)
    psr_mid  = val.probabilistic_sharpe_ratio(mid_eq,  benchmark_sharpe=benchmark_sr)
    psr_high = val.probabilistic_sharpe_ratio(high_eq, benchmark_sharpe=benchmark_sr)

    # All values in (0, 1)
    for res in (psr_low, psr_mid, psr_high):
        assert 0.0 < res.psr < 1.0, f"PSR={res.psr} not in (0,1)"

    # Monotonicity
    assert psr_low.psr < psr_mid.psr < psr_high.psr, (
        f"PSR not monotone: {psr_low.psr:.4f} < {psr_mid.psr:.4f} < {psr_high.psr:.4f}"
    )

    # NaN safety: constant equity
    flat = _equity_from_daily(np.zeros(200))
    res_flat = val.probabilistic_sharpe_ratio(flat, benchmark_sharpe=0.0)
    # Should return NaN PSR rather than crash
    assert math.isnan(res_flat.psr) or 0.0 <= res_flat.psr <= 1.0


# ---------------------------------------------------------------------------
# Test 3 — deflated_sharpe_ratio: penalizes many trials
# ---------------------------------------------------------------------------

def test_dsr_penalizes_many_trials():
    """DSR with more trial Sharpes lowers the adjusted benchmark, making DSR harder to pass.

    Concretely: a fixed observed equity curve should have lower DSR when the
    number of trial sharpes (or their max) is higher.
    """
    rng = np.random.default_rng(13)
    n = 600
    eq = _equity_from_daily(rng.normal(0.0006, 0.01, n))

    # Few trials: three mediocre sharpes
    trial_few = np.array([0.0, 0.1, 0.2])
    # Many trials: 50 sharpes spanning a wider range
    trial_many = rng.normal(0.5, 0.3, 50)

    dsr_few  = val.deflated_sharpe_ratio(eq, trial_sharpes=trial_few)
    dsr_many = val.deflated_sharpe_ratio(eq, trial_sharpes=trial_many)

    # Both in (0, 1)
    for res in (dsr_few, dsr_many):
        assert 0.0 < res.dsr < 1.0, f"DSR={res.dsr} not in (0,1)"

    # Many trials → higher expected-max benchmark → lower DSR
    assert dsr_many.dsr < dsr_few.dsr, (
        f"Expected DSR_many ({dsr_many.dsr:.4f}) < DSR_few ({dsr_few.dsr:.4f})"
    )

    # expected_max_sharpe should increase with more / higher trials
    assert dsr_many.expected_max_sharpe >= dsr_few.expected_max_sharpe - 1e-9


# ---------------------------------------------------------------------------
# Test 4 — newey_west_tstat: detects positive drift
# ---------------------------------------------------------------------------

def test_newey_west_tstat_positive_drift():
    """NW t-stat should be significantly positive for a strongly trending equity curve."""
    rng = np.random.default_rng(21)
    n = 800
    # Strong positive drift
    r_pos = rng.normal(0.001, 0.01, n)
    # Near-zero drift
    r_zero = rng.normal(0.0, 0.01, n)

    eq_pos  = _equity_from_daily(r_pos)
    eq_zero = _equity_from_daily(r_zero)

    res_pos  = val.newey_west_tstat(eq_pos)
    res_zero = val.newey_west_tstat(eq_zero)

    # Positive drift → positive t-stat
    assert res_pos.t_stat > 0, f"Expected positive t-stat, got {res_pos.t_stat}"

    # Strong drift → large t
    assert res_pos.t_stat > 2.0, (
        f"Expected t > 2 for strong drift series, got {res_pos.t_stat:.3f}"
    )

    # Zero drift → t-stat should be much smaller (in absolute terms)
    assert abs(res_zero.t_stat) < abs(res_pos.t_stat)

    # Lag is a non-negative integer
    assert isinstance(res_pos.lags, int) and res_pos.lags >= 0

    # NaN-safety: too-short series
    short = _equity_from_daily(np.array([0.01, 0.02]))
    res_short = val.newey_west_tstat(short)
    # Should not crash; t may be nan or a number
    assert res_short is not None


# ---------------------------------------------------------------------------
# Test 5 — bootstrap_sharpe_ci: brackets realized Sharpe and is deterministic
# ---------------------------------------------------------------------------

def test_stationary_bootstrap_ci_brackets_and_is_deterministic():
    """Stationary bootstrap CI should:
    1. Bracket the point-estimate Sharpe (ci_low < sharpe_point < ci_high).
    2. Be deterministic under the same seed.
    3. Be wider than zero (non-degenerate equity curve).
    """
    rng = np.random.default_rng(99)
    n = 500
    r = rng.normal(0.0006, 0.01, n)
    eq = _equity_from_daily(r)

    res_a = val.bootstrap_sharpe_ci(eq, n_boot=500, seed=0)
    res_b = val.bootstrap_sharpe_ci(eq, n_boot=500, seed=0)

    # Deterministic
    assert res_a.ci_low  == res_b.ci_low,  "CI low not deterministic"
    assert res_a.ci_high == res_b.ci_high, "CI high not deterministic"

    # CI brackets point estimate (with a small tolerance for edge cases)
    assert res_a.ci_low  <= res_a.sharpe_point + 1e-10
    assert res_a.ci_high >= res_a.sharpe_point - 1e-10

    # Non-degenerate
    assert res_a.ci_high > res_a.ci_low, "Bootstrap CI has zero width"

    # Different seed → different (but similar order-of-magnitude) results
    res_c = val.bootstrap_sharpe_ci(eq, n_boot=500, seed=1)
    assert res_c.ci_low != res_a.ci_low or res_c.ci_high != res_a.ci_high

    # NaN-safety: flat equity
    flat = _equity_from_daily(np.zeros(200))
    res_flat = val.bootstrap_sharpe_ci(flat, n_boot=100, seed=0)
    # Should not raise; CI may be nan or zero-width
    assert res_flat is not None
