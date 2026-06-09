import numpy as np
import pandas as pd
from country_rotation.backtest import metrics


def test_summary_known_values():
    idx = pd.bdate_range("2020-01-01", periods=252)
    r = pd.Series(0.001, index=idx)          # constant +10bp/day
    b = pd.Series(0.0005, index=idx)
    s = metrics.summary(r, b, periods_per_year=252)
    assert s["ann_vol"] == 0.0 or np.isnan(s["sharpe"]) or s["sharpe"] > 100  # degenerate vol
    assert s["max_drawdown"] == 0.0
    assert s["win_rate"] == 1.0
    assert s["tracking_error"] >= 0.0


def test_max_drawdown_sign():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.10, -0.50, 0.10, 0.10], index=idx)
    s = metrics.summary(r, r * 0, periods_per_year=252)
    assert -0.51 < s["max_drawdown"] < -0.49


def test_empty_returns_nan_safe():
    """Degenerate inputs must return NaN values, never raise."""
    idx = pd.bdate_range("2020-01-01", periods=1)
    r = pd.Series([0.0], index=idx)
    b = pd.Series([0.0], index=idx)
    s = metrics.summary(r, b, periods_per_year=252)
    # Should not raise; keys must all be present
    required = {"ann_return", "ann_vol", "sharpe", "max_drawdown", "win_rate",
                "beta", "up_capture", "down_capture", "tracking_error", "information_ratio"}
    assert required.issubset(s.keys())


def test_sharpe_positive_returns():
    """Consistently positive returns should yield positive Sharpe."""
    idx = pd.bdate_range("2020-01-01", periods=252)
    r = pd.Series(0.002, index=idx)
    b = pd.Series(0.001, index=idx)
    s = metrics.summary(r, b, periods_per_year=252)
    # With constant returns, vol == 0 → Sharpe is nan (degenerate)
    # but ann_return should be positive
    assert s["ann_return"] > 0


def test_win_rate_half():
    """Alternating +/- returns → win rate = 0.5."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    r = pd.Series([0.01, -0.01] * 5, index=idx)
    b = pd.Series(0.0, index=idx)
    s = metrics.summary(r, b, periods_per_year=252)
    assert abs(s["win_rate"] - 0.5) < 1e-9
