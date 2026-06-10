"""Tests for the benchmark-aware 'Cap_Tilt' construction (pre-registered
design, docs/research/segment_verdicts_2026-06-09.md "Next lever").

TDD: written first (red) before the Engine branch exists.

Cap_Tilt contract
-----------------
* ``Engine(scores, prices, cfg, base_weights=...)`` — cap-index base
  weights (dates x countries) plus/minus score tilts.
* Requires ``cfg.mode == 'active'`` and ``base_weights is not None``;
  ValueError at init otherwise.
* At each rebalance: base = last base_weights row at or before d_sel,
  restricted to valid (score+price) countries, renormalized to 1;
  top-N by selection signal get ``+ total_under / N``; bottom-N get
  ``- min(active_share/N, base_i)`` (long-only clip).  Sum stays 1.
* Existing branches (Equal / Risk_Parity, blend / active) are untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from country_rotation.backtest.engine import Engine
from country_rotation.config import BacktestConfig, load_config


# ---------------------------------------------------------------------------
# Inline fixtures (conftest-style: 10 countries + 'World' benchmark)
# ---------------------------------------------------------------------------

N_DAYS = 800
PERIODICITY = 21
COUNTRIES = [f"C{i}" for i in range(10)]


@pytest.fixture
def ct_prices() -> pd.DataFrame:
    """10 countries + 'World' benchmark, 800 business days."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-02", periods=N_DAYS)
    drifts = np.linspace(-0.0003, 0.0009, 10)
    cols = {}
    for k, name in enumerate(COUNTRIES):
        r = rng.normal(drifts[k], 0.012, N_DAYS)
        cols[name] = 100.0 * np.cumprod(1.0 + r)
    df = pd.DataFrame(cols, index=idx)
    df["World"] = df[COUNTRIES].mean(axis=1)
    return df


@pytest.fixture
def ct_scores(ct_prices) -> pd.DataFrame:
    """Smooth random scores in [0,1] for the 10 countries."""
    rng = np.random.default_rng(1)
    raw = rng.random((len(ct_prices), len(COUNTRIES)))
    df = pd.DataFrame(raw, index=ct_prices.index, columns=COUNTRIES)
    return df.rolling(21, min_periods=1).mean()


@pytest.fixture
def ct_base_weights(ct_prices) -> pd.DataFrame:
    """Stable cap-style base weights: softmax of a fixed vector, broadcast."""
    vec = np.linspace(-1.0, 1.5, len(COUNTRIES))
    w = np.exp(vec) / np.exp(vec).sum()
    return pd.DataFrame(
        np.tile(w, (len(ct_prices), 1)),
        index=ct_prices.index,
        columns=COUNTRIES,
    )


def _cfg(**kw) -> BacktestConfig:
    base = dict(
        selection_criteria="relative",
        relative_selection_score=3,
        weighting_method="Cap_Tilt",
        bmk="World",
        bmk_weight=0.0,
        mode="active",
        periodicity=PERIODICITY,
        transaction_cost_bps=2.0,
        active_share=0.3,
    )
    base.update(kw)
    return BacktestConfig(**base)


def _grid_dates(scores: pd.DataFrame, prices: pd.DataFrame, periodicity: int) -> list:
    """Replicate the engine's score-anchored reverse date grid."""
    grid = sorted(scores.index[::-periodicity])
    return sorted(prices.index.intersection(grid))


# ---------------------------------------------------------------------------
# 1. Full investment + long-only
# ---------------------------------------------------------------------------

def test_weights_sum_to_one_and_long_only(ct_prices, ct_scores, ct_base_weights):
    res = Engine(ct_scores, ct_prices, _cfg(), base_weights=ct_base_weights).run()
    hw = res.historical_weights
    assert len(hw) > 10
    sums = hw.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)
    assert (hw.to_numpy() >= -1e-12).all()


# ---------------------------------------------------------------------------
# 2. Top names overweight, bottom names underweight, middle at base
# ---------------------------------------------------------------------------

def test_top_overweight_bottom_underweight(ct_prices, ct_scores, ct_base_weights):
    cfg = _cfg()
    res = Engine(ct_scores, ct_prices, cfg, base_weights=ct_base_weights).run()
    hw = res.historical_weights

    dates = _grid_dates(ct_scores, ct_prices, cfg.periodicity)
    k = 5  # an arbitrary mid-sample rebalance
    d_sel, d_ret = dates[k], dates[k + 1]

    # Independently replicate the engine's relative signal at d_sel.
    idx = ct_scores.index.get_loc(d_sel)
    d_prev = ct_scores.index[idx - cfg.periodicity]
    signal = (ct_scores.loc[d_sel] - ct_scores.loc[d_prev]).dropna()

    base = ct_base_weights.loc[d_sel]
    base = base / base.sum()

    top = signal.nlargest(3).index.tolist()
    bottom = signal.drop(labels=top).nsmallest(3).index.tolist()
    middle = [c for c in COUNTRIES if c not in top and c not in bottom]

    w = hw.loc[d_ret]
    for c in top:
        assert w[c] > base[c], f"top name {c} not overweight"
    for c in bottom:
        assert w[c] < base[c] or np.isclose(w[c], 0.0, atol=1e-12), (
            f"bottom name {c} not underweight"
        )
    for c in middle:
        assert np.isclose(w[c], base[c], atol=1e-9), f"middle name {c} moved off base"


# ---------------------------------------------------------------------------
# 3. Long-only clip at base weight + sum conservation
# ---------------------------------------------------------------------------

def test_underweight_clipped_at_base(ct_prices):
    """A bottom name with base 0.001 < tilt_unit (0.1) clips to exactly 0;
    total realized underweight equals total overweight (sum conservation)."""
    idx = ct_prices.index
    # Deterministic monotonic scores: score change strictly increasing in i,
    # so top-3 = {C7, C8, C9} and bottom-3 = {C0, C1, C2} at every rebalance.
    t = np.arange(len(idx), dtype=float)
    scores = pd.DataFrame(
        {c: 1e-4 * i * t for i, c in enumerate(COUNTRIES)},
        index=idx,
    )
    # Base: C0 tiny (0.001), rest share the remainder equally.
    w = np.full(10, 0.999 / 9.0)
    w[0] = 0.001
    base_weights = pd.DataFrame(np.tile(w, (len(idx), 1)), index=idx, columns=COUNTRIES)

    cfg = _cfg()
    res = Engine(scores, ct_prices, cfg, base_weights=base_weights).run()
    hw = res.historical_weights

    base = pd.Series(w, index=COUNTRIES)
    top = ["C7", "C8", "C9"]
    bottom = ["C0", "C1", "C2"]

    for _, row in hw.iterrows():
        # C0 clipped: underweight = min(0.1, 0.001) = 0.001 -> weight 0
        assert row["C0"] >= 0.0
        assert np.isclose(row["C0"], 0.0, atol=1e-12)
        total_over = sum(row[c] - base[c] for c in top)
        total_under = sum(base[c] - row[c] for c in bottom)
        assert np.isclose(total_over, total_under, atol=1e-9)
        assert np.isclose(row.sum(), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 4. Init validation
# ---------------------------------------------------------------------------

def test_requires_active_mode_and_base(ct_prices, ct_scores, ct_base_weights):
    with pytest.raises(ValueError):
        Engine(ct_scores, ct_prices, _cfg(mode="blend", bmk_weight=0.5),
               base_weights=ct_base_weights)
    with pytest.raises(ValueError):
        Engine(ct_scores, ct_prices, _cfg(), base_weights=None)


def test_active_share_config_default():
    assert BacktestConfig().active_share == pytest.approx(0.30)
    cfg = load_config("configs/default.json")
    assert cfg.backtest.active_share == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# 5. No look-ahead
# ---------------------------------------------------------------------------

def test_no_lookahead(ct_prices, ct_scores, ct_base_weights):
    """Perturbing prices AND base weights strictly after t must not change
    historical weights at <= t."""
    cfg = _cfg()
    res1 = Engine(ct_scores, ct_prices, cfg, base_weights=ct_base_weights).run()

    cut = ct_prices.index[400]
    future = ct_prices.index > cut

    pert_prices = ct_prices.copy()
    pert_prices.loc[future] *= 3.0

    pert_bw = ct_base_weights.copy()
    flipped = pert_bw.iloc[0].to_numpy()[::-1]  # different valid weight vector
    pert_bw.loc[future] = np.tile(flipped, (int(future.sum()), 1))

    res2 = Engine(ct_scores, pert_prices, cfg, base_weights=pert_bw).run()

    w1 = res1.historical_weights.loc[res1.historical_weights.index <= cut]
    w2 = res2.historical_weights.loc[res2.historical_weights.index <= cut]
    pd.testing.assert_frame_equal(w1, w2)


# ---------------------------------------------------------------------------
# 6. Threading smoke: validation + protocols accept base_weights
# ---------------------------------------------------------------------------

def test_threading_smoke(ct_prices, ct_scores, ct_base_weights):
    from country_rotation.validation import protocols as proto
    from country_rotation.validation import scorecard as sc

    cfg = _cfg()
    grid = {"relative_selection_score": (3, 5)}

    report = sc.compute_validation(
        ct_scores, ct_prices, cfg, grid,
        n_boot=50, n_mc=5, n_folds=3, seed=0,
        basis="active", base_weights=ct_base_weights,
    )
    expected_keys = {"no_overfitting", "param_stable", "statistically_significant"}
    assert set(report.verdict.checks.keys()) == expected_keys
    for key, val in report.verdict.checks.items():
        assert isinstance(val, bool), f"checks[{key!r}] is {type(val)}, not bool"
    assert report.verdict.overall == all(report.verdict.checks.values())

    sweep = proto.parameter_sweep(
        ct_scores, ct_prices, cfg, grid,
        basis="active", base_weights=ct_base_weights,
    )
    sharpes = sweep.table["sharpe_ann"].to_numpy(dtype=float)
    assert len(sharpes) >= 2
    assert np.isfinite(sharpes).all()

    mc = proto.monte_carlo_null(
        ct_scores, ct_prices, cfg, n_sims=5, seed=0,
        basis="active", base_weights=ct_base_weights,
    )
    assert 0.0 < mc.p_value <= 1.0
    assert np.isfinite(mc.null_mean)
    assert np.isfinite(mc.actual_sharpe_ann)
