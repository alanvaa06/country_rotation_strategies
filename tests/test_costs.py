"""Per-country trading costs (Engine ``cost_bps``) + TCA module tests.

Engine vector-cost contract
---------------------------
* ``cost_bps=None``  -> byte-identical to the legacy flat behaviour
  (``cost = turnover * cfg.transaction_cost_bps / 1e4``).
* ``cost_bps`` given -> per-period cost = sum_i traded_notional_i * bps_i/1e4
  where traded_notional_i follows the legacy turnover decomposition:
  first period = full deployment ``w_i`` (no halving), afterwards
  ``|dw_i| / 2``, ignoring changes <= 1e-6, benchmark excluded.
  Reduces EXACTLY to the flat formula when bps_i is constant.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from country_rotation.backtest import tca
from country_rotation.backtest.engine import Engine, EngineResult
from country_rotation.config import BacktestConfig

COSTS_JSON = Path(__file__).resolve().parents[1] / "configs" / "costs.json"


def _cfg(**kw):
    base = dict(selection_criteria="relative", relative_selection_score=3,
                weighting_method="Equal", bmk="World", bmk_weight=0.5,
                mode="blend", periodicity=21, transaction_cost_bps=2.0)
    base.update(kw)
    return BacktestConfig(**base)


def _hand_traded_notional(weights: pd.DataFrame) -> pd.DataFrame:
    """Engine convention: first period = full deployment |w_i| (no halving),
    afterwards |dw_i|/2; changes <= 1e-6 ignored."""
    delta = weights.diff()
    delta.iloc[0] = weights.iloc[0]
    notional = delta.abs().where(delta.abs() > 1e-6, 0.0) / 2.0
    notional.iloc[0] = notional.iloc[0] * 2.0
    return notional


# ---------------------------------------------------------------------------
# Part 1 — Engine per-country costs
# ---------------------------------------------------------------------------

def test_flat_vector_identical_to_flat_config(synthetic_prices, synthetic_scores):
    """cost_bps with every value == cfg.transaction_cost_bps must reproduce
    the flat run exactly (the vector math reduces to turnover * bps/1e4)."""
    cfg = _cfg(transaction_cost_bps=10.0)
    flat = Engine(synthetic_scores, synthetic_prices, cfg).run()
    countries = [c for c in synthetic_scores.columns]
    vec = pd.Series(10.0, index=countries)
    vector = Engine(synthetic_scores, synthetic_prices, cfg, cost_bps=vec).run()
    pd.testing.assert_frame_equal(
        flat.period_results, vector.period_results, rtol=0.0, atol=1e-15
    )
    pd.testing.assert_frame_equal(flat.historical_weights, vector.historical_weights)


def test_doubling_traded_country_bps_increases_costs(synthetic_prices, synthetic_scores):
    cfg = _cfg(mode="active", bmk_weight=0.0, transaction_cost_bps=10.0)
    base = Engine(synthetic_scores, synthetic_prices, cfg).run()
    traded = base.historical_weights.diff().abs().sum()
    target = traded.idxmax()  # most-traded country
    assert traded[target] > 0

    countries = list(synthetic_scores.columns)
    vec = pd.Series(10.0, index=countries)
    flat_run = Engine(synthetic_scores, synthetic_prices, cfg, cost_bps=vec).run()
    vec2 = vec.copy()
    vec2[target] = 20.0
    bumped = Engine(synthetic_scores, synthetic_prices, cfg, cost_bps=vec2).run()

    assert (
        bumped.period_results["transaction_cost"].sum()
        > flat_run.period_results["transaction_cost"].sum()
    )


def test_missing_country_falls_back_to_flat_bps(synthetic_prices, synthetic_scores):
    """Countries absent from cost_bps are charged cfg.transaction_cost_bps."""
    cfg = _cfg(transaction_cost_bps=7.0)
    countries = list(synthetic_scores.columns)
    partial = pd.Series(15.0, index=countries[:5])
    explicit = pd.Series(
        [15.0] * 5 + [7.0] * (len(countries) - 5), index=countries
    )
    r_partial = Engine(synthetic_scores, synthetic_prices, cfg, cost_bps=partial).run()
    r_explicit = Engine(synthetic_scores, synthetic_prices, cfg, cost_bps=explicit).run()
    pd.testing.assert_frame_equal(
        r_partial.period_results, r_explicit.period_results, rtol=0.0, atol=1e-15
    )


def test_vector_cost_matches_hand_computation(synthetic_prices, synthetic_scores):
    """transaction_cost column == sum_i traded_notional_i * bps_i / 1e4
    recomputed from historical_weights (active mode: no benchmark column)."""
    cfg = _cfg(mode="active", bmk_weight=0.0, transaction_cost_bps=5.0)
    countries = list(synthetic_scores.columns)
    rng = np.random.default_rng(3)
    vec = pd.Series(rng.uniform(4.0, 30.0, len(countries)), index=countries)
    res = Engine(synthetic_scores, synthetic_prices, cfg, cost_bps=vec).run()

    notional = _hand_traded_notional(res.historical_weights)
    rates = vec.reindex(res.historical_weights.columns) / 1e4
    expected = notional.mul(rates, axis=1).sum(axis=1)
    assert np.allclose(
        res.period_results["transaction_cost"].to_numpy(),
        expected.to_numpy(), rtol=0.0, atol=1e-15,
    )


# ---------------------------------------------------------------------------
# Part 3 — TCA: fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hand_engine_result():
    """Hand-built EngineResult: 4 names rotating {A,B,C} <-> {B,C,D} equal
    weight every 21 business days, with a positive-drift active daily curve.

    Hand math: first-period turnover = 1.0 (full deployment), every later
    rebalance swaps one name: |dw| = 1/3 + 1/3 -> turnover = 1/3.
    """
    n = 757
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(7)
    bmk = pd.Series(rng.normal(0.0003, 0.010, n), index=idx)
    active = pd.Series(rng.normal(0.0006, 0.004, n), index=idx)
    port = bmk + active

    period_dates = idx[21::21]
    names = ["A", "B", "C", "D"]
    rows = []
    for k in range(len(period_dates)):
        held = ["A", "B", "C"] if k % 2 == 0 else ["B", "C", "D"]
        rows.append({c: (1.0 / 3.0 if c in held else 0.0) for c in names})
    weights = pd.DataFrame(rows, index=period_dates)

    turnover = [1.0] + [1.0 / 3.0] * (len(period_dates) - 1)
    period_results = pd.DataFrame(
        {
            "portfolio_return_gross": 0.010,
            "portfolio_return_net": 0.0098,
            "bmk_return": 0.008,
            "active_return": 0.002,
            "turnover": turnover,
            "transaction_cost": [t * 2.0e-4 for t in turnover],
        },
        index=period_dates,
    )
    return EngineResult(
        period_results=period_results,
        historical_weights=weights,
        daily_returns=port,
        daily_bmk_returns=bmk,
    )


@pytest.fixture
def flat_cost_model():
    """All-equal costs: spread 9 + commission 1 = 10 bps one-way."""
    return tca.CostModel(
        as_of="2026-06-10",
        commission_bps=1.0,
        spread_bps_by_country={c: 9.0 for c in ["A", "B", "C", "D"]},
        max_spread_bps=9.0,
        expense_ratio_bps={"DM": 50.0, "EM": 65.0, "World": 55.0},
        mgmt_fee_scenarios_bps=(0.0, 50.0),
    )


# ---------------------------------------------------------------------------
# Part 3 — CostModel loading
# ---------------------------------------------------------------------------

def test_load_cost_model_tiers_and_vector():
    model = tca.load_cost_model(COSTS_JSON)
    vec = model.cost_vector(["Japan", "China", "Indonesia"])
    # one_way_spread + commission(1):  dm_tier1 5+1, em_tier1 12+1, em_tier2 25+1
    assert vec["Japan"] == 6.0
    assert vec["China"] == 13.0
    assert vec["Indonesia"] == 26.0
    assert model.expense_ratio_bps["DM"] == 50.0
    assert model.expense_ratio_bps["EM"] == 65.0
    assert model.mgmt_fee_scenarios_bps == (0.0, 50.0)


def test_cost_vector_unknown_country_max_tier_and_warns():
    model = tca.load_cost_model(COSTS_JSON)
    with pytest.warns(UserWarning, match="Atlantis"):
        vec = model.cost_vector(["Atlantis"])
    assert vec["Atlantis"] == 26.0  # max tier spread (25) + commission (1)


def test_cost_vector_known_countries_no_warning():
    model = tca.load_cost_model(COSTS_JSON)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.cost_vector(["Japan", "Brazil", "Poland"])


# ---------------------------------------------------------------------------
# Part 3 — turnover_analysis
# ---------------------------------------------------------------------------

def test_turnover_matches_hand_computed(hand_engine_result):
    report = tca.turnover_analysis(hand_engine_result)
    expected = hand_engine_result.period_results["turnover"]
    assert np.allclose(report.per_rebalance.to_numpy(), expected.to_numpy(), atol=1e-12)
    assert report.per_rebalance.index.equals(expected.index)


def test_turnover_report_fields(hand_engine_result):
    report = tca.turnover_analysis(hand_engine_result)
    n_periods = len(hand_engine_result.period_results)

    assert report.avg_turnover == pytest.approx(report.per_rebalance.mean())
    assert report.max_turnover == pytest.approx(1.0)

    idx = hand_engine_result.daily_returns.index
    years = (idx[-1] - idx[0]).days / 365.25
    assert report.ann_one_way_turnover == pytest.approx(
        report.per_rebalance.sum() / years
    )

    # A and D carry the rotation; B and C only trade on deployment.
    top_two = list(report.country_avg_traded.index[:2])
    assert set(top_two) == {"A", "D"}
    assert (report.country_avg_traded.diff().dropna() <= 1e-12).all()  # descending

    # 3 adds on deployment, then one add + one drop per rebalance.
    assert int(report.name_changes.iloc[0]) == 3
    assert (report.name_changes.iloc[1:] == 2).all()
    assert report.total_name_changes == 3 + 2 * (n_periods - 1)


def test_turnover_matches_engine_column_on_real_run(synthetic_prices, synthetic_scores):
    cfg = _cfg(mode="active", bmk_weight=0.0)
    res = Engine(synthetic_scores, synthetic_prices, cfg).run()
    report = tca.turnover_analysis(res)
    assert np.allclose(
        report.per_rebalance.to_numpy(),
        res.period_results["turnover"].to_numpy(), atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Part 3 — cost_decomposition
# ---------------------------------------------------------------------------

def test_decomposition_flat_cost_equivalence(synthetic_prices, synthetic_scores):
    """All-equal model (9+1 bps) on a flat-10bps engine run: recomputed
    spread+commission cost must equal the engine's transaction_cost column."""
    cfg = _cfg(mode="active", bmk_weight=0.0, transaction_cost_bps=10.0)
    res = Engine(synthetic_scores, synthetic_prices, cfg).run()
    countries = list(res.historical_weights.columns)
    model = tca.CostModel(
        as_of="t", commission_bps=1.0,
        spread_bps_by_country={c: 9.0 for c in countries},
        max_spread_bps=9.0,
        expense_ratio_bps={"World": 55.0},
        mgmt_fee_scenarios_bps=(0.0,),
    )
    report = tca.cost_decomposition(res, model, "World")
    trading = report.per_period["spread_cost"] + report.per_period["commission_cost"]
    assert np.allclose(
        trading.to_numpy(),
        res.period_results["transaction_cost"].to_numpy(),
        rtol=0.0, atol=1e-15,
    )


def test_decomposition_per_period_and_ann_bps(hand_engine_result, flat_cost_model):
    report = tca.cost_decomposition(hand_engine_result, flat_cost_model, "DM")
    turnover = hand_engine_result.period_results["turnover"]

    # spread 9bps + commission 1bps on the hand turnover series
    assert np.allclose(
        report.per_period["spread_cost"].to_numpy(),
        (turnover * 9.0 / 1e4).to_numpy(), atol=1e-15,
    )
    assert np.allclose(
        report.per_period["commission_cost"].to_numpy(),
        (turnover * 1.0 / 1e4).to_numpy(), atol=1e-15,
    )
    # expense accrual annualises back to ~the configured expense ratio
    assert report.expense_ann_bps == pytest.approx(50.0, abs=1.5)
    assert report.mgmt_ann_bps[50.0] == pytest.approx(50.0, abs=1.5)
    assert report.mgmt_ann_bps[0.0] == pytest.approx(0.0, abs=1e-12)
    assert report.spread_ann_bps > 0.0
    assert report.commission_ann_bps == pytest.approx(report.spread_ann_bps / 9.0)


def test_decomposition_layers_ordered(hand_engine_result, flat_cost_model):
    """Cumulative cost layers: gross >= +spread >= +expense >= +mgmt(50)."""
    report = tca.cost_decomposition(hand_engine_result, flat_cost_model, "DM")
    irs = report.layer_irs
    assert report.gross_ir == pytest.approx(irs["gross"])
    assert irs["gross"] > 0  # fixture built with positive active drift
    assert irs["net_spread"] <= irs["gross"]
    assert irs["net_spread_expense"] <= irs["net_spread"]
    assert irs["net_mgmt_50bps"] <= irs["net_spread_expense"]
    # the 0-bps fee scenario adds no drag on top of the expense layer
    assert irs["net_mgmt_0bps"] == pytest.approx(irs["net_spread_expense"], abs=1e-12)


# ---------------------------------------------------------------------------
# Part 3 — breakeven_cost
# ---------------------------------------------------------------------------

def test_breakeven_positive_finite_and_consistent(hand_engine_result):
    res = hand_engine_result
    active = (res.daily_returns - res.daily_bmk_returns).dropna()
    gross_ir = tca.daily_ir(active)
    assert gross_ir > 0

    bps = tca.breakeven_cost(res, gross_ir)
    assert np.isfinite(bps)
    assert bps > 0

    # Re-run check: charge flat one-way `bps` on every rebalance -> IR ~ 0.
    cost = res.period_results["turnover"] * bps / 1e4
    net = active - cost.reindex(active.index).fillna(0.0)
    assert abs(tca.daily_ir(net)) < 0.05


# ---------------------------------------------------------------------------
# Part 4 — cost_bps threading: scorecard / protocols / report
# ---------------------------------------------------------------------------

def _cost_vector(columns) -> pd.Series:
    """Deterministic non-flat per-country cost vector for threading tests."""
    rng = np.random.default_rng(11)
    return pd.Series(rng.uniform(5.0, 30.0, len(columns)), index=list(columns))


def test_compute_validation_threads_cost_bps(synthetic_prices, synthetic_scores):
    """compute_validation with a cost vector returns a valid report and the
    absolute-basis stats differ from the flat-cost run (costs flow through
    net period returns)."""
    from country_rotation.validation import scorecard as sc

    prices = synthetic_prices.iloc[:400]
    scores = synthetic_scores.loc[prices.index]
    cfg = _cfg(mode="active", bmk_weight=0.0, transaction_cost_bps=2.0)
    grid = {"relative_selection_score": (3, 5)}
    vec = _cost_vector(scores.columns)

    report = sc.compute_validation(
        scores, prices, cfg, grid,
        n_boot=50, n_mc=5, n_folds=3, seed=0,
        basis="absolute", cost_bps=vec,
    )
    expected_keys = {"no_overfitting", "param_stable", "statistically_significant"}
    assert set(report.verdict.checks.keys()) == expected_keys
    for key, val in report.verdict.checks.items():
        assert isinstance(val, bool), f"checks[{key!r}] is {type(val)}, not bool"
    assert report.verdict.overall == all(report.verdict.checks.values())

    flat = sc.compute_validation(
        scores, prices, cfg, grid,
        n_boot=50, n_mc=5, n_folds=3, seed=0,
        basis="absolute",
    )
    # Costed absolute-basis Sharpe must differ from the flat 2bps run.
    assert report.sharpe.sharpe_ann != pytest.approx(flat.sharpe.sharpe_ann)


def test_sweep_and_mc_deterministic_with_cost_bps(synthetic_prices, synthetic_scores):
    """parameter_sweep / monte_carlo_null with cost_bps are deterministic
    across calls (same seed -> identical tables / p-values)."""
    from country_rotation.validation import protocols as proto

    prices = synthetic_prices.iloc[:400]
    scores = synthetic_scores.loc[prices.index]
    cfg = _cfg(mode="active", bmk_weight=0.0)
    grid = {"relative_selection_score": (3, 5)}
    vec = _cost_vector(scores.columns)

    s1 = proto.parameter_sweep(scores, prices, cfg, grid, cost_bps=vec)
    s2 = proto.parameter_sweep(scores, prices, cfg, grid, cost_bps=vec)
    pd.testing.assert_frame_equal(s1.table, s2.table)
    assert len(s1.table) >= 2
    assert np.isfinite(s1.table["sharpe_ann"].to_numpy(dtype=float)).all()

    m1 = proto.monte_carlo_null(scores, prices, cfg, n_sims=5, seed=0, cost_bps=vec)
    m2 = proto.monte_carlo_null(scores, prices, cfg, n_sims=5, seed=0, cost_bps=vec)
    assert m1.p_value == m2.p_value
    assert np.array_equal(m1.null_sharpes, m2.null_sharpes)
    assert 0.0 < m1.p_value <= 1.0

    wf = proto.walk_forward(scores, prices, cfg, grid, n_folds=3, cost_bps=vec)
    assert len(wf.fold_table) >= 1


def test_build_report_threads_cost_bps(synthetic_prices, synthetic_scores, tmp_path):
    """build_report accepts cost_bps and writes a complete HTML report."""
    from country_rotation.reporting.report import build_report

    prices = synthetic_prices.iloc[:400]
    scores = synthetic_scores.loc[prices.index]
    cfg = _cfg(mode="active", bmk_weight=0.0)
    out = tmp_path / "report_tca.html"

    info = build_report(
        scores, prices, cfg, str(out), cost_bps=_cost_vector(scores.columns)
    )
    assert out.exists()
    assert info["n_figures"] >= 3
    html = out.read_text(encoding="utf-8")
    assert "Performance" in html
