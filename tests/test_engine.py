import numpy as np
import pandas as pd
import pytest
from country_rotation.backtest.engine import Engine
from country_rotation.config import BacktestConfig


def _cfg(**kw):
    base = dict(selection_criteria="relative", relative_selection_score=3,
                weighting_method="Equal", bmk="World", bmk_weight=0.5,
                mode="blend", periodicity=21, transaction_cost_bps=2.0)
    base.update(kw)
    return BacktestConfig(**base)


def _grid_dates(scores, prices, periodicity):
    """Replicate the engine's score-anchored reverse date grid."""
    grid = sorted(scores.index[::-periodicity])
    return sorted(prices.index.intersection(grid))


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


# ---------------------------------------------------------------------------
# 'top_n_level' selection: canonical AMP-2013 rank-of-signal-level selection
# ---------------------------------------------------------------------------

def test_top_n_level_picks_highest_levels(synthetic_prices, synthetic_scores):
    """top_n_level selects the N highest score LEVELS at d_sel (manual nlargest)."""
    cfg = _cfg(selection_criteria="top_n_level", mode="active", bmk_weight=0.0)
    res = Engine(synthetic_scores, synthetic_prices, cfg).run()

    dates = _grid_dates(synthetic_scores, synthetic_prices, cfg.periodicity)
    for k in (3, 10, 20):
        d_sel, d_ret = dates[k], dates[k + 1]
        expected = set(
            synthetic_scores.loc[d_sel].dropna()
            .nlargest(cfg.relative_selection_score).index
        )
        w = res.historical_weights.loc[d_ret]
        selected = set(w[w > 0].index) - {"World"}
        assert selected == expected
        assert np.allclose(w[sorted(selected)], 1.0 / 3, atol=1e-12)


def test_top_n_level_differs_from_relative(synthetic_prices, synthetic_scores):
    """Level selection and change selection must not be the same portfolio."""
    res_lvl = Engine(synthetic_scores, synthetic_prices,
                     _cfg(selection_criteria="top_n_level")).run()
    res_rel = Engine(synthetic_scores, synthetic_prices,
                     _cfg(selection_criteria="relative")).run()
    assert not res_lvl.historical_weights.equals(res_rel.historical_weights)


def test_top_n_level_under_cap_tilt(synthetic_prices, synthetic_scores):
    """Cap_Tilt with top_n_level: weights sum 1, top-N by LEVEL overweight,
    bottom-N by the SAME level signal underweight."""
    countries = list(synthetic_scores.columns)
    base_vec = np.full(len(countries), 1.0 / len(countries))
    base_weights = pd.DataFrame(
        np.tile(base_vec, (len(synthetic_prices), 1)),
        index=synthetic_prices.index, columns=countries,
    )
    cfg = _cfg(selection_criteria="top_n_level", weighting_method="Cap_Tilt",
               mode="active", bmk_weight=0.0, active_share=0.3)
    res = Engine(synthetic_scores, synthetic_prices, cfg,
                 base_weights=base_weights).run()
    hw = res.historical_weights

    assert np.allclose(hw.sum(axis=1), 1.0, atol=1e-9)
    assert (hw.to_numpy() >= -1e-12).all()

    dates = _grid_dates(synthetic_scores, synthetic_prices, cfg.periodicity)
    d_sel, d_ret = dates[5], dates[6]
    level = synthetic_scores.loc[d_sel].dropna()
    top = level.nlargest(3).index.tolist()
    bottom = level.drop(labels=top).nsmallest(3).index.tolist()
    w = hw.loc[d_ret]
    base = 1.0 / len(countries)
    for c in top:
        assert w[c] > base, f"top name {c} not overweight"
    for c in bottom:
        assert w[c] < base, f"bottom name {c} not underweight"


def test_top_n_level_accepted_and_invalid_rejected(synthetic_prices, synthetic_scores):
    Engine(synthetic_scores, synthetic_prices, _cfg(selection_criteria="top_n_level"))
    with pytest.raises(ValueError, match="selection_criteria"):
        Engine(synthetic_scores, synthetic_prices, _cfg(selection_criteria="bogus"))


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
