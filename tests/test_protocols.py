"""Tests for country_rotation/validation/protocols.py (Task B2).

Covers: equity_curve, parameter_sweep + stability_summary, anchored
walk-forward (incl. the no-look-ahead perturbation test), and the
Monte-Carlo random-signal null (determinism + oracle power test).

Uses conftest fixtures ``synthetic_prices`` / ``synthetic_scores``,
sliced to short spans for speed.
"""
import math
from dataclasses import replace

import numpy as np
import pandas as pd

from country_rotation.config import BacktestConfig
from country_rotation.validation import protocols as proto


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_cfg(**overrides) -> BacktestConfig:
    """Fast active-mode relative-selection config used across tests."""
    base = BacktestConfig(
        selection_criteria="relative",
        relative_selection_score=4,
        weighting_method="Equal",
        bmk="World",
        bmk_weight=0.0,
        mode="active",
        periodicity=21,
        transaction_cost_bps=2.0,
    )
    return replace(base, **overrides) if overrides else base


# ---------------------------------------------------------------------------
# 1. equity_curve
# ---------------------------------------------------------------------------

def test_equity_curve_basic():
    """cumprod(1+r) with a prepended 1.0; empty input -> single 1.0."""
    r = pd.Series([0.1, -0.05])
    eq = proto.equity_curve(r)
    assert len(eq) == 3
    assert np.allclose(eq.to_numpy(dtype=float), [1.0, 1.1, 1.045])

    # Empty input is NaN-safe
    empty = proto.equity_curve(pd.Series(dtype=float))
    assert len(empty) == 1
    assert float(empty.iloc[0]) == 1.0

    # Datetime index: prepended point sits strictly before the first return
    idx = pd.bdate_range("2020-01-06", periods=3)
    r_dt = pd.Series([0.01, 0.02, -0.01], index=idx)
    eq_dt = proto.equity_curve(r_dt)
    assert len(eq_dt) == 4
    assert eq_dt.index[0] < idx[0]
    assert eq_dt.index.is_monotonic_increasing
    assert np.allclose(
        eq_dt.to_numpy(dtype=float),
        np.concatenate([[1.0], np.cumprod(1.0 + np.array([0.01, 0.02, -0.01]))]),
    )


# ---------------------------------------------------------------------------
# 2. parameter_sweep + stability_summary
# ---------------------------------------------------------------------------

def test_parameter_sweep_columns_and_rows(synthetic_prices, synthetic_scores):
    """1-D neighborhood sweep: dedup vs base, metric columns, finite sharpes."""
    px = synthetic_prices.iloc[:500]
    sc = synthetic_scores.iloc[:500]
    base = _base_cfg()  # periodicity 21, relative_selection_score 4
    grid = {"periodicity": (10, 21), "relative_selection_score": (3, 5)}

    sweep = proto.parameter_sweep(sc, px, base, grid)

    # base + periodicity 10 + rel 3 + rel 5 (periodicity 21 dedups into base)
    assert len(sweep.table) == 4
    for col in ("sharpe_daily", "sharpe_ann", "total_return", "max_drawdown",
                "periodicity", "relative_selection_score"):
        assert col in sweep.table.columns, f"missing column {col}"
    assert np.isfinite(sweep.table["sharpe_ann"].to_numpy(dtype=float)).all()
    assert isinstance(sweep.base_key, tuple)

    summary = proto.stability_summary(sweep, base)
    assert summary.n_trials == len(sweep.table)
    assert 0.0 <= summary.frac_positive <= 1.0
    for value in (summary.sharpe_mean, summary.sharpe_std, summary.sharpe_min,
                  summary.sharpe_max, summary.default_sharpe, summary.default_zscore):
        assert np.isfinite(value), f"non-finite stability field: {value}"
    # default row must be the base config's sharpe, not the median fallback
    base_mask = (
        (sweep.table["periodicity"] == base.periodicity)
        & (sweep.table["relative_selection_score"] == base.relative_selection_score)
    )
    assert summary.default_sharpe == float(sweep.table.loc[base_mask, "sharpe_ann"].iloc[0])


# ---------------------------------------------------------------------------
# 3. walk_forward — anchored, no look-ahead
# ---------------------------------------------------------------------------

def test_walk_forward_no_lookahead(synthetic_prices, synthetic_scores):
    """Fold-1 selection and IS stats must not change when data strictly
    after fold 1's OOS segment end is perturbed."""
    px = synthetic_prices.iloc[:500]
    sc = synthetic_scores.iloc[:500]
    base = _base_cfg()
    grid = {"relative_selection_score": (3, 5)}

    wf = proto.walk_forward(sc, px, base, grid, n_folds=3)
    ft = wf.fold_table

    assert 2 <= len(ft) <= 3
    # OOS segments are chronological and disjoint
    starts = list(ft["oos_start"])
    ends = list(ft["oos_end"])
    for i in range(1, len(ft)):
        assert starts[i] > starts[i - 1]
        assert starts[i] > ends[i - 1]
    assert int((ft["n_oos_periods"] >= 1).sum()) >= 2
    assert len(wf.oos_returns) > 0
    assert wf.oos_returns.index.is_monotonic_increasing
    assert 0.0 <= wf.frac_folds_oos_positive <= 1.0

    # Perturb prices strictly AFTER fold-1 oos_end
    oos_end_1 = ft.iloc[0]["oos_end"]
    px2 = px.copy()
    future = px2.index > oos_end_1
    assert future.any(), "perturbation window must be non-empty"
    px2.loc[future] = px2.loc[future] * 2.0

    wf2 = proto.walk_forward(sc, px2, base, grid, n_folds=3)
    row_a = ft.iloc[0]
    row_b = wf2.fold_table.iloc[0]
    assert row_a["params"] == row_b["params"]
    assert row_a["is_sharpe_daily"] == row_b["is_sharpe_daily"]
    assert row_a["oos_start"] == row_b["oos_start"]
    assert row_a["oos_end"] == row_b["oos_end"]
    # OOS run truncates at oos_end, so fold-1 OOS stats are untouched too
    assert row_a["n_oos_periods"] == row_b["n_oos_periods"]


# ---------------------------------------------------------------------------
# 4. monte_carlo_null — determinism, bounds
# ---------------------------------------------------------------------------

def test_monte_carlo_null_deterministic_and_bounded(synthetic_prices, synthetic_scores):
    px = synthetic_prices.iloc[:400]
    sc = synthetic_scores.iloc[:400]
    cfg = _base_cfg(relative_selection_score=5)

    mc_a = proto.monte_carlo_null(sc, px, cfg, n_sims=15, seed=3)
    mc_b = proto.monte_carlo_null(sc, px, cfg, n_sims=15, seed=3)

    assert mc_a.null_sharpes.size == 15
    assert mc_a.n_sims == 15
    assert np.array_equal(mc_a.null_sharpes, mc_b.null_sharpes)
    assert mc_a.p_value == mc_b.p_value
    assert 0.0 < mc_a.p_value <= 1.0
    assert np.isfinite(mc_a.null_mean)
    assert np.isfinite(mc_a.null_q95)


# ---------------------------------------------------------------------------
# 5. monte_carlo_null — oracle signal sits in the top tail
# ---------------------------------------------------------------------------

def test_monte_carlo_oracle_signal_beats_null(synthetic_prices):
    """Forward-return-rank oracle (as in tests/test_ic.py) must beat almost
    every random-signal null: p <= 2/16 with add-one smoothing, n_sims=15."""
    px = synthetic_prices
    countries = [c for c in px.columns if c != "World"]
    pxc = px[countries]
    fwd = pxc.shift(-21) / pxc - 1.0
    scores = fwd.rank(axis=1, pct=True).fillna(0.5)

    cfg = _base_cfg(relative_selection_score=5)
    mc = proto.monte_carlo_null(scores, px, cfg, n_sims=15, seed=0)

    assert not math.isnan(mc.actual_sharpe_ann)
    assert mc.p_value <= 2.0 / 16.0


# ---------------------------------------------------------------------------
# 6. parameter_sweep — active (excess-over-benchmark) basis
# ---------------------------------------------------------------------------

def test_parameter_sweep_active_basis(synthetic_prices, synthetic_scores):
    """basis='active' yields finite sharpes and differs from absolute basis."""
    px = synthetic_prices.iloc[:500]
    sc = synthetic_scores.iloc[:500]
    base = _base_cfg()
    grid = {"relative_selection_score": (3, 5)}

    sweep_abs = proto.parameter_sweep(sc, px, base, grid, basis="absolute")
    sweep_act = proto.parameter_sweep(sc, px, base, grid, basis="active")

    # Same shape (basis only changes the metric values, not the rows/cols)
    assert list(sweep_abs.table.columns) == list(sweep_act.table.columns)
    assert len(sweep_abs.table) == len(sweep_act.table)
    assert sweep_act.base_key == sweep_abs.base_key

    # Active sharpes are finite
    act_sharpe = sweep_act.table["sharpe_ann"].to_numpy(dtype=float)
    assert np.isfinite(act_sharpe).all()

    # Default basis is absolute and matches the explicit absolute call
    sweep_default = proto.parameter_sweep(sc, px, base, grid)
    assert np.allclose(
        sweep_default.table["sharpe_daily"].to_numpy(dtype=float),
        sweep_abs.table["sharpe_daily"].to_numpy(dtype=float),
    )

    # Active basis removes the benchmark drift: at least one trial differs
    assert not np.allclose(
        sweep_abs.table["sharpe_daily"].to_numpy(dtype=float),
        sweep_act.table["sharpe_daily"].to_numpy(dtype=float),
    )

    # Deterministic: same inputs reproduce the table exactly
    sweep_act2 = proto.parameter_sweep(sc, px, base, grid, basis="active")
    assert np.allclose(
        sweep_act.table["sharpe_daily"].to_numpy(dtype=float),
        sweep_act2.table["sharpe_daily"].to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# 7. monte_carlo_null — active basis
# ---------------------------------------------------------------------------

def test_monte_carlo_null_active_basis(synthetic_prices, synthetic_scores):
    """basis='active' gives a valid p in (0,1], is deterministic, and differs
    from the absolute basis on the same inputs."""
    px = synthetic_prices.iloc[:400]
    sc = synthetic_scores.iloc[:400]
    cfg = _base_cfg(relative_selection_score=5)

    mc_act_a = proto.monte_carlo_null(sc, px, cfg, n_sims=15, seed=3, basis="active")
    mc_act_b = proto.monte_carlo_null(sc, px, cfg, n_sims=15, seed=3, basis="active")
    mc_abs = proto.monte_carlo_null(sc, px, cfg, n_sims=15, seed=3, basis="absolute")

    # Deterministic under the same seed/basis
    assert np.array_equal(mc_act_a.null_sharpes, mc_act_b.null_sharpes)
    assert mc_act_a.p_value == mc_act_b.p_value

    # Valid bounded p-value, finite summary stats
    assert 0.0 < mc_act_a.p_value <= 1.0
    assert np.isfinite(mc_act_a.null_mean)
    assert np.isfinite(mc_act_a.null_q95)
    assert not math.isnan(mc_act_a.actual_sharpe_ann)

    # Default basis is absolute
    mc_default = proto.monte_carlo_null(sc, px, cfg, n_sims=15, seed=3)
    assert mc_default.actual_sharpe_ann == mc_abs.actual_sharpe_ann
    assert np.array_equal(mc_default.null_sharpes, mc_abs.null_sharpes)

    # Active actual sharpe differs from absolute (benchmark drift removed)
    assert mc_act_a.actual_sharpe_ann != mc_abs.actual_sharpe_ann
