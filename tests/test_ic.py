import numpy as np
import pandas as pd
from country_rotation.backtest import ic


def test_perfect_signal_gives_ic_one(synthetic_prices):
    countries = [c for c in synthetic_prices.columns if c != "World"]
    px = synthetic_prices[countries]
    periodicity = 21
    fwd = px.shift(-periodicity) / px - 1.0
    scores = fwd.rank(axis=1, pct=True)      # oracle: score = future-return rank
    out = ic.information_coefficient(scores, px, periodicity, method="absolute")
    assert out["IC"].dropna().mean() > 0.95


def test_random_signal_ic_near_zero(synthetic_prices, synthetic_scores):
    countries = [c for c in synthetic_prices.columns if c != "World"]
    out = ic.information_coefficient(synthetic_scores, synthetic_prices[countries], 21, method="absolute")
    stats = ic.ic_stats(out["IC"])
    assert abs(stats["mean_ic"]) < 0.15
    assert set(stats) >= {"mean_ic", "icir", "t_stat", "hit_rate"}


def test_relative_method_no_crash(synthetic_prices, synthetic_scores):
    """Relative method (score diff) should run without error and return valid DataFrame."""
    countries = [c for c in synthetic_prices.columns if c != "World"]
    out = ic.information_coefficient(synthetic_scores, synthetic_prices[countries], 21, method="relative")
    assert "IC" in out.columns
    # Should have fewer rows than absolute (first period dropped for diff)
    out_abs = ic.information_coefficient(synthetic_scores, synthetic_prices[countries], 21, method="absolute")
    assert len(out) <= len(out_abs)


def test_nonpositive_start_price_excluded():
    """Countries with p0 <= 0 get a NaN forward return and are dropped from the IC pair."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    cols = ["A", "B", "C", "D"]
    rng = np.random.default_rng(7)
    px = pd.DataFrame(100.0 + rng.normal(0, 1, (10, 4)), index=idx, columns=cols)
    px.loc[idx[4], "D"] = 0.0  # zero start price at a grid date (periodicity=5 → rows 4, 9)
    scores = pd.DataFrame(rng.random((10, 4)), index=idx, columns=cols)
    out = ic.information_coefficient(scores, px, periodicity=5, method="absolute")
    # Pair (idx[4] -> idx[9]) survives with only A, B, C (D dropped, not Inf/-1)
    assert len(out) == 1
    assert int(out["n_countries"].iloc[0]) == 3


def test_ic_stats_t_stat_formula():
    """Hand-built IC series: verify t_stat = mean/std * sqrt(n)."""
    rng = np.random.default_rng(42)
    vals = pd.Series(rng.normal(0.1, 0.2, 30))
    stats = ic.ic_stats(vals)
    n = len(vals)
    expected_t = vals.mean() / vals.std() * np.sqrt(n)
    assert abs(stats["t_stat"] - expected_t) < 1e-9
    assert abs(stats["icir"] - vals.mean() / vals.std()) < 1e-9


def test_ic_stats_keys():
    """ic_stats must return all required keys."""
    vals = pd.Series([0.1, -0.2, 0.3, 0.05, -0.1])
    stats = ic.ic_stats(vals)
    required = {"mean_ic", "median_ic", "std_ic", "t_stat", "icir", "hit_rate"}
    assert required.issubset(stats.keys())
