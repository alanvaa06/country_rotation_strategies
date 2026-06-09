"""New Engine must reproduce legacy Backtest period returns exactly (blend mode).

Legacy ``run_backtest()`` returns the dict produced by ``get_results()``
(backtest.py ~889-917).  Key name mapping:

    Legacy key              Engine column / attribute
    ----------------------  ----------------------------------------
    'returns'               period_results['portfolio_return_gross']
    'returns_net'           period_results['portfolio_return_net']
    'benchmark_returns'     period_results['bmk_return']
    'active_return'         period_results['active_return']
    'turnover'              period_results['turnover']
    'transaction_costs'     period_results['transaction_cost']
    'daily_returns'         engine_result.daily_returns
    'daily_benchmark_returns' engine_result.daily_bmk_returns
"""
import os

# Prevent matplotlib from trying to open a display (legacy backtest imports plt).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest

from backtest import Backtest               # legacy root module
from country_rotation.backtest.engine import Engine
from country_rotation.config import BacktestConfig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_legacy(synthetic_scores, synthetic_prices, selection, weighting, bmkw, period):
    return Backtest(
        normalized_score=synthetic_scores,
        prices=synthetic_prices,
        selection_criteria=selection,
        absolute_selection_score=0.75,
        relative_selection_score=5,
        weighting_method=weighting,
        bmk="World",
        bmk_weight=bmkw,
        periodicity=period,
        transaction_cost_bps=2.0,
    )


def _make_engine(synthetic_scores, synthetic_prices, selection, weighting, bmkw, period):
    cfg = BacktestConfig(
        selection_criteria=selection,
        absolute_selection_score=0.75,
        relative_selection_score=5,
        weighting_method=weighting,
        risk_parity_lookback=60,   # match legacy Backtest default (60, not 120)
        bmk="World",
        bmk_weight=bmkw,
        mode="blend",
        periodicity=period,
        transaction_cost_bps=2.0,
    )
    return Engine(synthetic_scores, synthetic_prices, cfg)


def _assert_series_equal(legacy_series, engine_series, label: str, atol: float = 1e-9):
    """Compare two array-likes element-wise and report the first divergence."""
    a = np.asarray(legacy_series, dtype=float)
    b = np.asarray(engine_series, dtype=float)
    assert a.shape == b.shape, (
        f"[{label}] shape mismatch: legacy={a.shape} engine={b.shape}"
    )
    if not np.allclose(a, b, atol=atol, equal_nan=True):
        diff = np.abs(a - b)
        idx = int(np.nanargmax(diff))
        max_diff = diff[idx]
        pytest.fail(
            f"[{label}] max |diff| = {max_diff:.3e} at index {idx}: "
            f"legacy={a[idx]:.10f}, engine={b[idx]:.10f}"
        )


# ---------------------------------------------------------------------------
# parametrized period-level parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("selection,weighting,bmkw,period", [
    ("relative", "Equal",        0.5, 21),
    ("relative", "Equal",        0.0, 63),
    ("absolute", "Equal",        0.3, 21),
    ("relative", "Risk_Parity",  0.5, 21),
])
def test_period_return_parity(synthetic_prices, synthetic_scores,
                               selection, weighting, bmkw, period):
    """All period-level metrics must match legacy Backtest exactly."""
    legacy = _make_legacy(synthetic_scores, synthetic_prices,
                          selection, weighting, bmkw, period)
    legacy_res = legacy.run_backtest()

    new_res = _make_engine(synthetic_scores, synthetic_prices,
                           selection, weighting, bmkw, period).run()

    pr = new_res.period_results

    # --- gross returns ---
    _assert_series_equal(
        legacy_res["returns"],
        pr["portfolio_return_gross"],
        label=f"gross [{selection},{weighting},bmkw={bmkw},p={period}]",
    )

    # --- net returns ---
    _assert_series_equal(
        legacy_res["returns_net"],
        pr["portfolio_return_net"],
        label=f"net [{selection},{weighting},bmkw={bmkw},p={period}]",
    )

    # --- benchmark returns ---
    _assert_series_equal(
        legacy_res["benchmark_returns"],
        pr["bmk_return"],
        label=f"bmk [{selection},{weighting},bmkw={bmkw},p={period}]",
    )

    # --- active returns ---
    _assert_series_equal(
        legacy_res["active_return"],
        pr["active_return"],
        label=f"active [{selection},{weighting},bmkw={bmkw},p={period}]",
    )

    # --- turnover ---
    _assert_series_equal(
        legacy_res["turnover"],
        pr["turnover"],
        label=f"turnover [{selection},{weighting},bmkw={bmkw},p={period}]",
    )

    # --- transaction costs ---
    _assert_series_equal(
        legacy_res["transaction_costs"],
        pr["transaction_cost"],
        label=f"tc [{selection},{weighting},bmkw={bmkw},p={period}]",
    )


# ---------------------------------------------------------------------------
# daily curve parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("selection,weighting,bmkw,period", [
    ("relative", "Equal",        0.5, 21),
    ("relative", "Equal",        0.0, 63),
    ("absolute", "Equal",        0.3, 21),
    ("relative", "Risk_Parity",  0.5, 21),
])
def test_daily_curve_parity(synthetic_prices, synthetic_scores,
                             selection, weighting, bmkw, period):
    """Engine daily curves must match legacy get_daily_returns() output."""
    legacy = _make_legacy(synthetic_scores, synthetic_prices,
                          selection, weighting, bmkw, period)
    legacy_res = legacy.run_backtest()
    # legacy stores daily_returns as an attribute set in run_backtest (~270-271)
    legacy_daily_port = legacy_res["daily_returns"]
    legacy_daily_bmk  = legacy_res["daily_benchmark_returns"]

    new_res = _make_engine(synthetic_scores, synthetic_prices,
                           selection, weighting, bmkw, period).run()

    assert new_res.daily_returns is not None, "Engine returned no daily_returns"
    assert new_res.daily_bmk_returns is not None, "Engine returned no daily_bmk_returns"

    _assert_series_equal(
        legacy_daily_port,
        new_res.daily_returns,
        label=f"daily_port [{selection},{weighting},bmkw={bmkw},p={period}]",
    )

    _assert_series_equal(
        legacy_daily_bmk,
        new_res.daily_bmk_returns,
        label=f"daily_bmk [{selection},{weighting},bmkw={bmkw},p={period}]",
    )
