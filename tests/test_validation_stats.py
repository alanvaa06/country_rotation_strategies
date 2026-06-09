"""Tests for country_rotation/validation/statistics.py (Task B1).

Tests — one per statistic plus canonical-value and property tests.
All equity-based tests use the _equity_from_daily helper to convert
daily returns to an equity curve.
"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

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


def test_sharpe_ann_equals_daily_times_sqrt252():
    """sharpe_ann must equal sharpe_daily * sqrt(252) for any valid series."""
    rng = np.random.default_rng(77)
    r = rng.normal(0.0005, 0.01, 500)
    eq = _equity_from_daily(r)
    res = val.sharpe_significance(eq)

    assert not math.isnan(res.sharpe_daily), "sharpe_daily should not be NaN"
    assert not math.isnan(res.sharpe_ann), "sharpe_ann should not be NaN"
    assert abs(res.sharpe_ann - res.sharpe_daily * math.sqrt(252)) < 1e-12, (
        f"sharpe_ann {res.sharpe_ann} != sharpe_daily*sqrt(252) "
        f"{res.sharpe_daily * math.sqrt(252)}"
    )

    # NaN propagation: flat equity → sharpe_ann should also be NaN
    flat = _equity_from_daily(np.zeros(100))
    res_flat = val.sharpe_significance(flat)
    assert math.isnan(res_flat.sharpe_ann), "sharpe_ann should be NaN when sharpe_daily is NaN"


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

    # PSRResult uses 'kurtosis' field (non-excess, Normal=3), not 'excess_kurtosis'
    assert hasattr(psr_mid, "kurtosis"), "PSRResult must have 'kurtosis' field"
    assert not hasattr(psr_mid, "excess_kurtosis"), (
        "PSRResult must NOT have 'excess_kurtosis' field (renamed to 'kurtosis')"
    )


def test_psr_canonical_value():
    """Verify PSR against hand-computed canonical value.

    Uses a known synthetic series with computable moments:
    returns = [0.01, -0.005, 0.02, 0.0, 0.015] * 200  (n=1000)

    We compute γ3 (bias=False), γ4 (fisher=False, bias=False), SR (ddof=1)
    manually via numpy/scipy and verify PSR equals
    Φ(SR * sqrt(n-1) / sqrt(1 − γ3·SR + ((γ4−1)/4)·SR²)) within 1e-12.
    """
    base = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    returns = np.tile(base, 200)  # n = 1000
    n = len(returns)
    assert n == 1000

    # Reference moments computed the same way as the implementation
    sr = float(np.mean(returns)) / float(np.std(returns, ddof=1))
    g3 = float(scipy_stats.skew(returns, bias=False))
    g4 = float(scipy_stats.kurtosis(returns, fisher=False, bias=False))

    variance_term = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2
    variance_term = max(variance_term, 1e-12)

    z_expected = sr * math.sqrt(n - 1) / math.sqrt(variance_term)
    psr_expected = float(scipy_stats.norm.cdf(z_expected))

    # Build equity curve from the synthetic returns
    eq = _equity_from_daily(returns)
    result = val.probabilistic_sharpe_ratio(eq, benchmark_sharpe=0.0)

    assert not math.isnan(result.psr), "PSR should not be NaN for well-formed series"
    assert abs(result.psr - psr_expected) < 1e-12, (
        f"PSR {result.psr:.15f} != expected {psr_expected:.15f}"
    )


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


def test_dsr_properties():
    """DSR/expected_max_sharpe direction and edge-case properties.

    Constructs two trial sets with the same sample variance so that increasing
    N unambiguously pushes expected_max_sharpe up (more trials = higher benchmark).
    Also verifies N=1 → expected_max = 0.0.
    """
    rng = np.random.default_rng(55)
    n = 800
    eq = _equity_from_daily(rng.normal(0.0008, 0.01, n))

    # Build two trial sets with identical empirical std but different N
    base_trials = rng.normal(0.3, 0.2, 200)
    trials_5   = base_trials[:5]
    trials_200 = base_trials  # all 200

    dsr_5   = val.deflated_sharpe_ratio(eq, trial_sharpes=trials_5)
    dsr_200 = val.deflated_sharpe_ratio(eq, trial_sharpes=trials_200)

    # More trials → higher expected_max_sharpe
    assert dsr_200.expected_max_sharpe > dsr_5.expected_max_sharpe, (
        f"expected_max with N=200 ({dsr_200.expected_max_sharpe:.4f}) should exceed "
        f"N=5 ({dsr_5.expected_max_sharpe:.4f})"
    )

    # More trials → lower DSR (harder benchmark to beat)
    assert dsr_200.dsr < dsr_5.dsr, (
        f"DSR with N=200 ({dsr_200.dsr:.4f}) should be < N=5 ({dsr_5.dsr:.4f})"
    )

    # N=1 → expected_max = 0.0
    result_n1 = val.deflated_sharpe_ratio(eq, trial_sharpes=np.array([0.5]))
    assert result_n1.expected_max_sharpe == 0.0, (
        f"N=1 expected_max should be 0.0, got {result_n1.expected_max_sharpe}"
    )


# ---------------------------------------------------------------------------
# Test 4 — newey_west_tstat: takes return series, detects positive drift
# ---------------------------------------------------------------------------

def test_newey_west_tstat_positive_drift():
    """NW t-stat should be significantly positive for a strong positive drift series.

    newey_west_tstat now takes a pd.Series of RETURNS (not an equity curve).
    """
    rng = np.random.default_rng(21)
    n = 3000  # large n for reliable t > 2
    # Strong positive drift
    r_pos = pd.Series(rng.normal(0.0006, 0.01, n))
    # Near-zero drift
    r_zero = pd.Series(rng.normal(0.0, 0.01, n))

    res_pos  = val.newey_west_tstat(r_pos)
    res_zero = val.newey_west_tstat(r_zero)

    # Positive drift → positive t-stat
    assert res_pos.t_stat > 0, f"Expected positive t-stat, got {res_pos.t_stat}"

    # Strong drift → t > 2
    assert res_pos.t_stat > 2.0, (
        f"Expected t > 2 for strong drift series (n={n}), got {res_pos.t_stat:.3f}"
    )

    # Zero drift → |t| should be small (< 2.5 with high probability for n=3000)
    assert abs(res_zero.t_stat) < 2.5, (
        f"Expected |t| < 2.5 for zero-mean noise, got {abs(res_zero.t_stat):.3f}"
    )

    # Lag is a non-negative integer
    assert isinstance(res_pos.lags, int) and res_pos.lags >= 0

    # Auto lag: minimum 1 for return series input
    assert res_pos.lags >= 1, f"Auto lag should be >= 1, got {res_pos.lags}"

    # NaN-safety: too-short series
    short = pd.Series([0.01, 0.02])
    res_short = val.newey_west_tstat(short)
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
