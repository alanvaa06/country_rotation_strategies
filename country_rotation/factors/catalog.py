# country_rotation/factors/catalog.py
"""Factor registry: every usable factor with category and direction.

Direction: +1 = higher is better, -1 = lower is better.
exploratory=True: no documented country-level literature support; held to
stricter selection thresholds (spec 2026-06-09 §6 Stage 1).

Corrections vs legacy FactorTesting.py lists:
- ROE/Fwd_ROE/Return_Capital -> Profitability (legacy had ROE in Valuation).
- Raw levels (Price, GDP, M2, Market_Cap, EV, Revenue, Debt, Equity,
  Liabilities, Assets, EPS, Earnings, EBIT, EBITDA, Ten_Year, SI, SI_Ratio)
  are NOT factors: levels are not cross-country comparable.
- Momentum_12_1 / Momentum_6_1 added (strongest documented country factor).
- RollingVol -> Quality, direction -1 (low-risk).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorSpec:
    name: str
    category: str
    direction: int
    exploratory: bool = False


CATALOG: tuple = (
    # Valuation — multiples (lower better)
    FactorSpec("PE", "Valuation", -1), FactorSpec("Fwd_PE", "Valuation", -1),
    FactorSpec("PB", "Valuation", -1), FactorSpec("PS", "Valuation", -1),
    FactorSpec("Fwd_PS", "Valuation", -1), FactorSpec("PCF", "Valuation", -1),
    FactorSpec("Fwd_PCF", "Valuation", -1), FactorSpec("EV_EBIT", "Valuation", -1),
    FactorSpec("EV_EBITDA", "Valuation", -1), FactorSpec("Fwd_EV_EBITDA", "Valuation", -1),
    # Valuation — yields & spreads (higher better)
    FactorSpec("EarningsYieldTTM", "Valuation", 1), FactorSpec("EarningsYieldFWD", "Valuation", 1),
    FactorSpec("CashFlowYieldTTM", "Valuation", 1), FactorSpec("CashFlowYieldFWD", "Valuation", 1),
    FactorSpec("DVD", "Valuation", 1), FactorSpec("Fwd_DVD", "Valuation", 1),
    FactorSpec("EarningsYieldTTMSpread", "Valuation", 1), FactorSpec("EarningsYieldFWDSpread", "Valuation", 1),
    FactorSpec("CashFlowYieldTTMSpread", "Valuation", 1), FactorSpec("CashFlowYieldFWDSpread", "Valuation", 1),
    FactorSpec("DvdYieldTTMSpread", "Valuation", 1), FactorSpec("DvdYieldFWDSpread", "Valuation", 1),
    # Quality
    FactorSpec("Debt_to_Equity", "Quality", -1), FactorSpec("Net_Debt_Ebitda", "Quality", -1),
    FactorSpec("AssetsEquity", "Quality", -1), FactorSpec("CF", "Quality", 1),
    FactorSpec("FwdCF", "Quality", 1), FactorSpec("RollingVol", "Quality", -1),
    # Profitability
    FactorSpec("ROE", "Profitability", 1), FactorSpec("Fwd_ROE", "Profitability", 1),
    FactorSpec("Return_Capital", "Profitability", 1),
    FactorSpec("EbitMargin", "Profitability", 1), FactorSpec("EbitdaMargin", "Profitability", 1),
    FactorSpec("NetMargin", "Profitability", 1), FactorSpec("FwdEBITDAMargin", "Profitability", 1),
    FactorSpec("FwdNetMargin", "Profitability", 1),
    FactorSpec("ConsensusSalesGrowth", "Profitability", 1, exploratory=True),
    FactorSpec("ConsensusEbitdaGrowth", "Profitability", 1, exploratory=True),
    FactorSpec("ConsensusEarningsGrowth", "Profitability", 1, exploratory=True),
    FactorSpec("ConsensusCashFlowGrowth", "Profitability", 1, exploratory=True),
    # Momentum
    FactorSpec("Momentum_12_1", "Momentum", 1), FactorSpec("Momentum_6_1", "Momentum", 1),
    FactorSpec("RollingEarnings", "Momentum", 1), FactorSpec("FwdRollingEarnings", "Momentum", 1),
    FactorSpec("CumFlow", "Momentum", 1, exploratory=True),
)

_BY_NAME = {f.name: f for f in CATALOG}


def get_spec(name: str) -> FactorSpec:
    return _BY_NAME[name]


def by_category(category: str) -> tuple:
    return tuple(f for f in CATALOG if f.category == category)
