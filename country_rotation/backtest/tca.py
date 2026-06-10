"""
Transaction cost analysis (TCA): pure post-trade analytics on EngineResult.

Design contract
---------------
* Pure functions over frozen dataclasses — the ONLY IO is
  :func:`load_cost_model` (reads ``configs/costs.json``).
* All trade math reuses the engine's vector-cost convention exactly
  (engine ``_transaction_cost`` / legacy turnover decomposition):
  first rebalance = full deployment ``w_i`` (no halving), afterwards
  ``|dw_i| / 2``, weight changes <= 1e-6 ignored.
* ``historical_weights`` from blend-mode results include the benchmark
  column; pass ``bmk=...`` to :func:`turnover_analysis` /
  :func:`cost_decomposition` to exclude it (active-mode results carry no
  benchmark column, so the default ``None`` is correct there).

Cost layers (cumulative, applied to the ACTIVE daily series)
------------------------------------------------------------
gross -> +spread/commission (charged on rebalance days)
      -> +ETF expense drag  (annual bps, accrued daily)
      -> +management fee    (per scenario, accrued daily)
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from country_rotation.backtest.engine import EngineResult

_TRADE_EPS = 1e-6           # engine threshold: ignore |dw| below this
_CAL_DAYS_PER_YEAR = 365.25
_TRADING_DAYS_PER_YEAR = 252.0


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """Per-country one-way implementation cost assumptions.

    ``cost_vector`` returns spread + commission in bps; unknown countries
    are charged the most expensive tier (conservative) with a warning.
    """

    as_of: str
    commission_bps: float
    spread_bps_by_country: Mapping[str, float]
    max_spread_bps: float
    expense_ratio_bps: Mapping[str, float]   # segment -> annual bps
    mgmt_fee_scenarios_bps: tuple            # strategy-level annual fee tiers

    def cost_vector(self, countries: Sequence[str]) -> pd.Series:
        """One-way trading cost in bps per country (spread + commission)."""
        unknown = [c for c in countries if c not in self.spread_bps_by_country]
        if unknown:
            warnings.warn(
                f"CostModel has no tier for {unknown}; falling back to the "
                f"max tier spread ({self.max_spread_bps:g} bps one-way).",
                UserWarning,
                stacklevel=2,
            )
        values = [
            self.spread_bps_by_country.get(c, self.max_spread_bps)
            + self.commission_bps
            for c in countries
        ]
        return pd.Series(values, index=list(countries), dtype=float)


def load_cost_model(path: str | Path) -> CostModel:
    """Load ``configs/costs.json`` into a frozen :class:`CostModel`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    spread_by_country: dict = {}
    for tier in raw["tiers"].values():
        tier_bps = float(tier["one_way_spread_bps"])
        for country in tier["countries"]:
            spread_by_country[country] = tier_bps
    return CostModel(
        as_of=str(raw["as_of"]),
        commission_bps=float(raw["commission_bps"]),
        spread_bps_by_country=MappingProxyType(spread_by_country),
        max_spread_bps=max(
            float(t["one_way_spread_bps"]) for t in raw["tiers"].values()
        ),
        expense_ratio_bps=MappingProxyType(
            {k: float(v) for k, v in raw["expense_ratio_bps"].items()}
        ),
        mgmt_fee_scenarios_bps=tuple(
            float(x) for x in raw["mgmt_fee_scenarios_bps"]
        ),
    )


# ---------------------------------------------------------------------------
# shared trade math (engine convention)
# ---------------------------------------------------------------------------

def _traded_notional(weights: pd.DataFrame) -> pd.DataFrame:
    """Per-country one-way traded notional per rebalance (engine convention).

    First row = full deployment ``|w_i|`` (legacy first-period turnover
    counts the whole sleeve once, no halving); afterwards ``|dw_i| / 2``;
    changes <= 1e-6 ignored — identical to ``Engine._transaction_cost``.
    """
    if weights.empty:
        return weights.copy()
    delta = weights.diff()
    delta.iloc[0] = weights.iloc[0]
    notional = delta.abs().where(delta.abs() > _TRADE_EPS, 0.0) / 2.0
    notional.iloc[0] = notional.iloc[0] * 2.0
    return notional


def _drop_bmk(weights: pd.DataFrame, bmk: Optional[str]) -> pd.DataFrame:
    if bmk is not None and bmk in weights.columns:
        return weights.drop(columns=[bmk])
    return weights


def _years_spanned(engine_result: EngineResult) -> float:
    """Calendar years covered (daily index preferred, period index fallback)."""
    if engine_result.daily_returns is not None and len(engine_result.daily_returns) > 1:
        idx = engine_result.daily_returns.index
    else:
        idx = engine_result.period_results.index
        if len(idx) < 2:
            return np.nan
    return (idx[-1] - idx[0]).days / _CAL_DAYS_PER_YEAR


def _fsum(values) -> float:
    """Exactly-rounded sum (``math.fsum``), NaN-skipping like pandas.

    numpy/pandas reductions use pairwise summation whose grouping can vary
    with memory alignment, producing 1-ULP run-to-run jitter in scalars —
    enough to break the byte-determinism contract of JSON artifacts derived
    from them. fsum is exactly rounded, hence bit-stable.
    """
    arr = np.asarray(values, dtype=float)
    return math.fsum(arr[np.isfinite(arr)])


def _fmean_fstd(values) -> tuple[float, float]:
    """Bit-stable (mean, sample std) via exactly-rounded sums."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return np.nan, np.nan
    mean = math.fsum(arr) / n
    var = math.fsum((arr - mean) ** 2) / (n - 1)
    return mean, math.sqrt(var)


def daily_ir(active_daily: pd.Series) -> float:
    """Annualized information ratio of a daily active-return series:
    ``mean / std * sqrt(252)``.  NaN on degenerate input."""
    clean = active_daily.dropna()
    if len(clean) < 2:
        return np.nan
    mean, std = _fmean_fstd(clean)
    if std == 0.0 or not np.isfinite(std):
        return np.nan
    return float(mean / std * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _active_daily(engine_result: EngineResult) -> Optional[pd.Series]:
    if engine_result.daily_returns is None or engine_result.daily_bmk_returns is None:
        return None
    return (engine_result.daily_returns - engine_result.daily_bmk_returns).dropna()


# ---------------------------------------------------------------------------
# turnover analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TurnoverReport:
    """Per-rebalance turnover diagnostics (engine one-way convention)."""

    per_rebalance: pd.Series         # one-way turnover per rebalance
    ann_one_way_turnover: float      # sum(turnover) / calendar years
    avg_turnover: float
    max_turnover: float
    country_avg_traded: pd.Series    # avg one-way traded notional, descending
    name_changes: pd.Series          # holdings entering + leaving per rebalance
    total_name_changes: int


def turnover_analysis(
    engine_result: EngineResult, bmk: Optional[str] = None
) -> TurnoverReport:
    """Recompute turnover from ``historical_weights`` (engine convention).

    For engine-produced results this reproduces
    ``period_results['turnover']`` exactly; pass ``bmk`` for blend-mode
    results so the benchmark column is excluded as the engine does.
    """
    weights = _drop_bmk(engine_result.historical_weights, bmk)
    notional = _traded_notional(weights)
    per_rebalance = notional.sum(axis=1)

    years = _years_spanned(engine_result)
    # round(…, 10): the engine's per-period turnover scalars carry 1-ULP
    # run-to-run jitter (alignment-dependent numpy reductions; the engine is
    # parity-locked so it cannot adopt fsum) — quantizing at 1e-10 makes the
    # derived artifact scalars bit-stable while losing nothing economic.
    ann_turnover = (
        round(_fsum(per_rebalance) / years, 10) if years and years > 0 else np.nan
    )

    held = weights.abs() > _TRADE_EPS
    flips = held.astype(int).diff()
    if len(flips) > 0:
        flips.iloc[0] = held.iloc[0].astype(int)
    name_changes = flips.abs().sum(axis=1).astype(int)

    return TurnoverReport(
        per_rebalance=per_rebalance,
        ann_one_way_turnover=ann_turnover,
        avg_turnover=float(per_rebalance.mean()) if len(per_rebalance) else np.nan,
        max_turnover=float(per_rebalance.max()) if len(per_rebalance) else np.nan,
        country_avg_traded=notional.mean().sort_values(ascending=False),
        name_changes=name_changes,
        total_name_changes=int(name_changes.sum()),
    )


# ---------------------------------------------------------------------------
# cost decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostReport:
    """Layered implementation-cost decomposition of an EngineResult.

    ``layer_irs`` keys: ``gross``, ``net_spread``, ``net_spread_expense``,
    then ``net_mgmt_{scenario:g}bps`` per fee scenario — each layer adds
    one cumulative drag to the ACTIVE daily series.
    """

    segment: str
    per_period: pd.DataFrame         # spread_cost, commission_cost,
                                     # expense_drag, mgmt_drag_{s}bps ...
    spread_ann_bps: float
    commission_ann_bps: float
    expense_ann_bps: float
    mgmt_ann_bps: Mapping[float, float]   # scenario bps -> realized ann bps
    gross_ir: float
    layer_irs: Mapping[str, float]


def _period_year_fractions(engine_result: EngineResult) -> pd.Series:
    """Calendar-year fraction covered by each period (for fee accrual).

    The first period's start is the daily curve's first date (the engine's
    first selection date) when available, otherwise the median gap.
    """
    idx = engine_result.period_results.index
    gaps = idx.to_series().diff().dt.days.astype(float)
    if engine_result.daily_returns is not None and len(engine_result.daily_returns):
        gaps.iloc[0] = float((idx[0] - engine_result.daily_returns.index[0]).days)
    else:
        gaps.iloc[0] = float(gaps.median()) if len(gaps) > 1 else np.nan
    return gaps / _CAL_DAYS_PER_YEAR


def cost_decomposition(
    engine_result: EngineResult,
    cost_model: CostModel,
    segment: str,
    bmk: Optional[str] = None,
) -> CostReport:
    """Spread/commission/expense/fee decomposition + layered active IRs.

    Spread and commission costs are recomputed from ``historical_weights``
    diffs with the engine's vector math (per-country traded notional times
    that country's bps), so on an engine run priced with the same vector
    they reproduce ``period_results['transaction_cost']`` exactly.
    """
    weights = _drop_bmk(engine_result.historical_weights, bmk)
    notional = _traded_notional(weights)

    total_rate = cost_model.cost_vector(list(weights.columns))
    spread_rate = (total_rate - cost_model.commission_bps) / 1e4
    commission_rate = cost_model.commission_bps / 1e4

    spread_cost = notional.mul(spread_rate, axis=1).sum(axis=1)
    commission_cost = (notional * commission_rate).sum(axis=1)

    expense_bps = float(cost_model.expense_ratio_bps[segment])
    year_frac = _period_year_fractions(engine_result)
    expense_drag = expense_bps / 1e4 * year_frac

    per_period = pd.DataFrame(
        {
            "spread_cost": spread_cost,
            "commission_cost": commission_cost,
            "expense_drag": expense_drag,
        }
    )
    mgmt_drags: dict = {}
    for scenario in cost_model.mgmt_fee_scenarios_bps:
        drag = scenario / 1e4 * year_frac
        mgmt_drags[scenario] = drag
        per_period[f"mgmt_drag_{scenario:g}bps"] = drag

    years = _years_spanned(engine_result)
    def _ann_bps(total_cost: float) -> float:
        # round(…, 10): bit-stability at the artifact boundary (see
        # turnover_analysis).
        return (
            round(float(total_cost / years * 1e4), 10)
            if years and years > 0 else np.nan
        )

    mgmt_ann_bps = {
        scenario: _ann_bps(_fsum(drag)) for scenario, drag in mgmt_drags.items()
    }

    # ----- layered IRs on the ACTIVE daily curve -----------------------
    active = _active_daily(engine_result)
    layer_irs: dict = {}
    if active is None or len(active) < 2:
        layer_irs["gross"] = np.nan
        layer_irs["net_spread"] = np.nan
        layer_irs["net_spread_expense"] = np.nan
        for scenario in cost_model.mgmt_fee_scenarios_bps:
            layer_irs[f"net_mgmt_{scenario:g}bps"] = np.nan
    else:
        trading_cost_daily = (
            (spread_cost + commission_cost).reindex(active.index).fillna(0.0)
        )
        net_spread = active - trading_cost_daily
        expense_daily = expense_bps / _TRADING_DAYS_PER_YEAR / 1e4
        net_spread_expense = net_spread - expense_daily

        layer_irs["gross"] = daily_ir(active)
        layer_irs["net_spread"] = daily_ir(net_spread)
        layer_irs["net_spread_expense"] = daily_ir(net_spread_expense)
        for scenario in cost_model.mgmt_fee_scenarios_bps:
            mgmt_daily = scenario / _TRADING_DAYS_PER_YEAR / 1e4
            layer_irs[f"net_mgmt_{scenario:g}bps"] = daily_ir(
                net_spread_expense - mgmt_daily
            )

    return CostReport(
        segment=segment,
        per_period=per_period,
        spread_ann_bps=_ann_bps(_fsum(spread_cost)),
        commission_ann_bps=_ann_bps(_fsum(commission_cost)),
        expense_ann_bps=_ann_bps(_fsum(expense_drag)),
        mgmt_ann_bps=MappingProxyType(mgmt_ann_bps),
        gross_ir=layer_irs["gross"],
        layer_irs=MappingProxyType(layer_irs),
    )


# ---------------------------------------------------------------------------
# breakeven cost
# ---------------------------------------------------------------------------

def breakeven_cost(engine_result: EngineResult, base_ir_gross: float) -> float:
    """Flat one-way bps level at which the active IR hits zero.

    Linear solve ``IR(bps) ~ base_ir_gross - k * bps`` with the slope from
    realized turnover: one flat bp charged on every rebalance lowers the
    daily active mean by ``sum(turnover) / n_days / 1e4``, so
    ``k = sum(turnover) * sqrt(252) / (n_days * std_daily * 1e4)`` and the
    breakeven is ``base_ir_gross / k`` (std change ignored — second order).
    """
    active = _active_daily(engine_result)
    if active is None or len(active) < 2:
        raise ValueError("breakeven_cost requires daily return curves")
    _, std_daily = _fmean_fstd(active)
    total_turnover = _fsum(engine_result.period_results["turnover"])
    if not np.isfinite(base_ir_gross):
        raise ValueError("base_ir_gross must be finite")
    if std_daily <= 0.0 or total_turnover <= 0.0:
        raise ValueError("breakeven undefined: zero turnover or zero active vol")

    slope_per_bp = (
        total_turnover
        * np.sqrt(_TRADING_DAYS_PER_YEAR)
        / (len(active) * std_daily * 1e4)
    )
    # round(…, 10): engine period turnover carries 1-ULP run-to-run jitter
    # (alignment-dependent reductions in the parity-locked engine);
    # quantizing keeps the artifact byte-deterministic.
    return round(float(base_ir_gross / slope_per_bp), 10)
