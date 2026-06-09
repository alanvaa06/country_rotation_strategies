import numpy as np
import pandas as pd
from country_rotation.backtest.engine import Engine
from country_rotation.config import BacktestConfig


def _cfg(**kw):
    base = dict(selection_criteria="relative", relative_selection_score=3,
                weighting_method="Equal", bmk="World", bmk_weight=0.5,
                mode="blend", periodicity=21, transaction_cost_bps=2.0)
    base.update(kw)
    return BacktestConfig(**base)


def test_blend_weights_sum_to_one(synthetic_prices, synthetic_scores):
    res = Engine(synthetic_scores, synthetic_prices, _cfg()).run()
    sums = res.historical_weights.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)
    assert (res.historical_weights["World"].iloc[1:] == 0.5).all()


def test_equal_weighting_among_selected(synthetic_prices, synthetic_scores):
    res = Engine(synthetic_scores, synthetic_prices, _cfg(bmk_weight=0.0)).run()
    w = res.historical_weights.drop(columns="World", errors="ignore")
    nonzero = w.iloc[5][w.iloc[5] > 0]
    assert len(nonzero) == 3
    assert np.allclose(nonzero, 1.0 / 3, atol=1e-9)


def test_tc_reduces_returns(synthetic_prices, synthetic_scores):
    r0 = Engine(synthetic_scores, synthetic_prices, _cfg(transaction_cost_bps=0.0)).run()
    r1 = Engine(synthetic_scores, synthetic_prices, _cfg(transaction_cost_bps=50.0)).run()
    assert r1.period_results["portfolio_return_net"].sum() < r0.period_results["portfolio_return_net"].sum()


def test_active_mode_reports_benchmark_relative(synthetic_prices, synthetic_scores):
    res = Engine(synthetic_scores, synthetic_prices, _cfg(mode="active", bmk_weight=0.0)).run()
    assert "active_return" in res.period_results.columns
    assert "World" not in res.historical_weights.columns or (res.historical_weights["World"] == 0).all()


def test_no_lookahead_in_engine(synthetic_prices, synthetic_scores):
    """Perturbing prices after date t must not change weights chosen at <= t."""
    res1 = Engine(synthetic_scores, synthetic_prices, _cfg()).run()
    cut = synthetic_prices.index[400]
    pert = synthetic_prices.copy()
    pert.loc[pert.index > cut] *= 3.0
    res2 = Engine(synthetic_scores, pert, _cfg()).run()
    w1 = res1.historical_weights.loc[res1.historical_weights.index <= cut]
    w2 = res2.historical_weights.loc[res2.historical_weights.index <= cut]
    pd.testing.assert_frame_equal(w1, w2)
