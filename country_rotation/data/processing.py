"""Derived-metric computations — pure functions with no I/O.

All inputs are ``dict[str, pd.DataFrame]`` (metric-keyed).  Every function
returns a **new** dict; the input dict and its contained DataFrames are never
mutated.

Public API
----------
add_consensus_growth      -- PS/Fwd ratios -> implied consensus growth
add_yields                -- earnings / cash-flow yields (1/multiple)
add_spreads               -- yield minus Ten_Year bond
add_reconstructions       -- absolute earnings/revenue/CF from multiples × price
add_margins               -- profit margins from reconstructed fundamentals
add_rolling_stats         -- rolling earnings revision, vol, cumulative flows
add_leverage              -- Assets / Equity
add_regional_aggregates   -- GDP/M2/Market_Cap sums + Ten_Year weighted averages
add_price_momentum        -- 12-1 and 6-1 price momentum signals
run_processing            -- chain all steps in legacy order, momentum last
"""
from __future__ import annotations

import warnings
from math import sqrt
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _safe_div(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    """Element-wise num/den with nonpositive/zero denominators -> NaN (D8).

    All multiples (PE, PS, PCF, EV_EBITDA), Revenue, and Equity should be
    positive in valid data.  Masking non-positive values prevents economically
    meaningless results from raw division.
    """
    den_masked = den.where(den > 0)
    return num / den_masked


def _warn_missing(func_name: str, missing: list[str]) -> None:
    warnings.warn(
        f"{func_name}: required inputs missing — {missing}. Skipping.",
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Step 3 — Consensus growth
# ---------------------------------------------------------------------------

def add_consensus_growth(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute consensus growth proxies from TTM vs Forward multiples.

    Formulas (ported from ProcessData.py ~408-418):
        ConsensusSalesGrowth     = PS / Fwd_PS - 1
        ConsensusEbitdaGrowth    = EV_EBITDA / Fwd_EV_EBITDA - 1
        ConsensusEarningsGrowth  = PE / Fwd_PE - 1
        ConsensusCashFlowGrowth  = PCF / Fwd_PCF - 1

    Note: These are ratio-of-multiples formulas, not direct yield divisions,
    so they do NOT go through ``_safe_div`` — the denominator is a forward
    multiple that can be negative in crisis periods.  Following the legacy
    code exactly for these formulas.
    """
    out = dict(dfs)

    if "PS" in out and "Fwd_PS" in out:
        out["ConsensusSalesGrowth"] = out["PS"] / out["Fwd_PS"] - 1

    if "EV_EBITDA" in out and "Fwd_EV_EBITDA" in out:
        out["ConsensusEbitdaGrowth"] = out["EV_EBITDA"] / out["Fwd_EV_EBITDA"] - 1

    if "PE" in out and "Fwd_PE" in out:
        out["ConsensusEarningsGrowth"] = out["PE"] / out["Fwd_PE"] - 1

    if "PCF" in out and "Fwd_PCF" in out:
        out["ConsensusCashFlowGrowth"] = out["PCF"] / out["Fwd_PCF"] - 1

    return out


# ---------------------------------------------------------------------------
# Step 4 — Yields
# ---------------------------------------------------------------------------

def add_yields(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute earnings and cash-flow yields (1 / multiple).

    Formulas (ported from ProcessData.py ~425-435):
        EarningsYieldTTM  = 1 / PE
        EarningsYieldFWD  = 1 / Fwd_PE
        CashFlowYieldTTM  = 1 / PCF
        CashFlowYieldFWD  = 1 / Fwd_PCF

    Uses ``_safe_div``: negative/zero multiples produce NaN (D8 fix).
    """
    out = dict(dfs)
    ones = pd.DataFrame  # alias for clarity — we create a scalar-1 df per call

    def _inv(key: str) -> Optional[pd.DataFrame]:
        if key in out:
            return _safe_div(
                pd.DataFrame(np.ones(out[key].shape), index=out[key].index, columns=out[key].columns),
                out[key],
            )
        return None

    ey_ttm = _inv("PE")
    if ey_ttm is not None:
        out["EarningsYieldTTM"] = ey_ttm

    ey_fwd = _inv("Fwd_PE")
    if ey_fwd is not None:
        out["EarningsYieldFWD"] = ey_fwd

    cf_ttm = _inv("PCF")
    if cf_ttm is not None:
        out["CashFlowYieldTTM"] = cf_ttm

    cf_fwd = _inv("Fwd_PCF")
    if cf_fwd is not None:
        out["CashFlowYieldFWD"] = cf_fwd

    return out


# ---------------------------------------------------------------------------
# Step 5 — Spreads
# ---------------------------------------------------------------------------

def add_spreads(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute yield spreads versus the 10-year bond.

    Formulas (ported from ProcessData.py ~442-459):
        EarningsYieldTTMSpread  = EarningsYieldTTM - Ten_Year
        EarningsYieldFWDSpread  = EarningsYieldFWD - Ten_Year
        CashFlowYieldTTMSpread  = CashFlowYieldTTM - Ten_Year
        CashFlowYieldFWDSpread  = CashFlowYieldFWD - Ten_Year
        DvdYieldTTMSpread       = DVD - Ten_Year
        DvdYieldFWDSpread       = Fwd_DVD - Ten_Year

    Uses DataFrame.sub(..., fill_value=0) to replicate legacy behavior:
    when a column is missing in Ten_Year the yield acts as if the bond rate
    were zero (i.e., spread = full yield).
    """
    if "Ten_Year" not in dfs:
        return dfs

    out = dict(dfs)
    ten_yr = out["Ten_Year"]

    _spread_pairs = [
        ("EarningsYieldTTM",  "EarningsYieldTTMSpread"),
        ("EarningsYieldFWD",  "EarningsYieldFWDSpread"),
        ("CashFlowYieldTTM",  "CashFlowYieldTTMSpread"),
        ("CashFlowYieldFWD",  "CashFlowYieldFWDSpread"),
        ("DVD",               "DvdYieldTTMSpread"),
        ("Fwd_DVD",           "DvdYieldFWDSpread"),
    ]

    for src_key, dst_key in _spread_pairs:
        if src_key in out:
            out[dst_key] = out[src_key].sub(ten_yr, fill_value=0)

    return out


# ---------------------------------------------------------------------------
# Step 6 — Reconstructions
# ---------------------------------------------------------------------------

def add_reconstructions(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Reconstruct absolute fundamentals from price × yield.

    Formulas (ported from ProcessData.py ~466-482):
        Earnings    = (1/PE)      * Price
        FwdEarnings = EarningsYieldFWD * Price
        FwdRevenue  = (1/Fwd_PS)  * Price
        FwdEBITDA   = (1/Fwd_EV_EBITDA) * EV
        CF          = (1/PCF)     * Price
        FwdCF       = (1/Fwd_PCF) * Price

    All inverse-multiple divisions go through ``_safe_div`` (D8 fix).
    """
    out = dict(dfs)

    def _inv_key(key: str) -> Optional[pd.DataFrame]:
        if key not in out:
            return None
        df = out[key]
        ones = pd.DataFrame(
            np.ones(df.shape), index=df.index, columns=df.columns
        )
        return _safe_div(ones, df)

    if "PE" in out and "Price" in out:
        inv_pe = _inv_key("PE")
        out["Earnings"] = inv_pe * out["Price"]

    # FwdEarnings uses already-computed EarningsYieldFWD if available;
    # otherwise fall back to computing inline from Fwd_PE.
    if "EarningsYieldFWD" in out and "Price" in out:
        out["FwdEarnings"] = out["EarningsYieldFWD"] * out["Price"]
    elif "Fwd_PE" in out and "Price" in out:
        inv_fwd_pe = _inv_key("Fwd_PE")
        out["FwdEarnings"] = inv_fwd_pe * out["Price"]

    if "Fwd_PS" in out and "Price" in out:
        inv_fwd_ps = _inv_key("Fwd_PS")
        out["FwdRevenue"] = inv_fwd_ps * out["Price"]

    if "Fwd_EV_EBITDA" in out and "EV" in out:
        inv_fwd_ev = _inv_key("Fwd_EV_EBITDA")
        out["FwdEBITDA"] = inv_fwd_ev * out["EV"]

    if "PCF" in out and "Price" in out:
        inv_pcf = _inv_key("PCF")
        out["CF"] = inv_pcf * out["Price"]

    if "Fwd_PCF" in out and "Price" in out:
        inv_fwd_pcf = _inv_key("Fwd_PCF")
        out["FwdCF"] = inv_fwd_pcf * out["Price"]

    return out


# ---------------------------------------------------------------------------
# Step 7 — Margins
# ---------------------------------------------------------------------------

def add_margins(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute profit margins from reconstructed fundamentals.

    Formulas (ported from ProcessData.py ~489-503):
        EbitMargin      = EBIT     / Revenue
        EbitdaMargin    = EBITDA   / Revenue
        NetMargin       = Earnings / Revenue
        FwdEBITDAMargin = FwdEBITDA / Revenue
        FwdNetMargin    = FwdEarnings / Revenue

    Revenue denominator goes through ``_safe_div`` (D8 fix).
    """
    if "Revenue" not in dfs:
        return dfs

    out = dict(dfs)
    rev = out["Revenue"]

    _margin_pairs = [
        ("EBIT",        "EbitMargin"),
        ("EBITDA",      "EbitdaMargin"),
        ("Earnings",    "NetMargin"),
        ("FwdEBITDA",   "FwdEBITDAMargin"),
        ("FwdEarnings", "FwdNetMargin"),
    ]

    for src_key, dst_key in _margin_pairs:
        if src_key in out:
            out[dst_key] = _safe_div(out[src_key], rev)

    return out


# ---------------------------------------------------------------------------
# Step 8 — Rolling statistics
# ---------------------------------------------------------------------------

def add_rolling_stats(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute rolling time-series statistics.

    Formulas (ported from ProcessData.py ~510-520):
        RollingEarnings    = Earnings.pct_change(63).rolling(21).mean()
        FwdRollingEarnings = FwdEarnings.pct_change(63).rolling(21).mean()
        RollingVol         = Price.pct_change().rolling(63).std() * sqrt(252)
        CumFlow            = Flows.rolling(63).sum()
    """
    out = dict(dfs)

    if "Earnings" in out:
        out["RollingEarnings"] = out["Earnings"].pct_change(63).rolling(21).mean()

    if "FwdEarnings" in out:
        out["FwdRollingEarnings"] = out["FwdEarnings"].pct_change(63).rolling(21).mean()

    if "Price" in out:
        out["RollingVol"] = out["Price"].pct_change().rolling(63).std() * sqrt(252)

    if "Flows" in out:
        out["CumFlow"] = out["Flows"].rolling(63).sum()

    return out


# ---------------------------------------------------------------------------
# Step 9 — Leverage
# ---------------------------------------------------------------------------

def add_leverage(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute balance-sheet leverage ratio.

    Formula (ported from ProcessData.py ~527-528):
        AssetsEquity = Assets / Equity

    Equity goes through ``_safe_div`` (D8 fix): non-positive equity -> NaN.
    """
    if "Assets" not in dfs or "Equity" not in dfs:
        return dfs

    out = dict(dfs)
    out["AssetsEquity"] = _safe_div(out["Assets"], out["Equity"])
    return out


# ---------------------------------------------------------------------------
# Step 2 — Regional aggregates
# ---------------------------------------------------------------------------

def add_regional_aggregates(
    dfs: dict[str, pd.DataFrame],
    classification: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute regional / segment aggregate columns for GDP, M2, Market_Cap, Ten_Year.

    Ported from ProcessData.py ~354-401.

    Parameters
    ----------
    dfs:
        Metric-keyed dict.  Modified keys: GDP, M2, Market_Cap, Ten_Year
        (new aggregate columns are appended; Ten_Year is also divided by 100
        before computing weighted averages).
    classification:
        DataFrame with columns Segment (EM/DM) and Region (Asia/Europe/LatAm).
        Index = country names.

    Returns
    -------
    dict[str, pd.DataFrame]
        Same keys as input; affected DataFrames have extra aggregate columns.
    """
    out = dict(dfs)

    # Build country lists from classification
    def _countries(col: str, val: str) -> list[str]:
        if col not in classification.columns:
            return []
        return classification[classification[col] == val].index.tolist()

    em_countries = _countries("Segment", "EM")
    dm_countries = _countries("Segment", "DM")
    asia_countries = _countries("Region", "Asia")
    europe_countries = _countries("Region", "Europe")
    latam_countries = _countries("Region", "LatAm")
    world_indices = classification.index.tolist()

    # -- GDP: EM + DM sums --
    if "GDP" in out:
        gdp = out["GDP"].copy()
        em_cols = [c for c in em_countries if c in gdp.columns]
        dm_cols = [c for c in dm_countries if c in gdp.columns]
        if em_cols:
            gdp["EM"] = gdp[em_cols].sum(axis=1)
        if dm_cols:
            gdp["DM"] = gdp[dm_cols].sum(axis=1)
        out["GDP"] = gdp

    # -- M2: DM/EM/Europe/Asia/LatAm/World sums --
    if "M2" in out:
        m2 = out["M2"].copy()
        _agg_specs = [
            ("DM", dm_countries),
            ("EM", em_countries),
            ("Europe", europe_countries),
            ("Asia", asia_countries),
            ("LatAm", latam_countries),
            ("World", world_indices),
        ]
        for label, countries in _agg_specs:
            cols = [c for c in countries if c in m2.columns]
            if cols:
                m2[label] = m2[cols].sum(axis=1)
        out["M2"] = m2

    # -- Market_Cap: DM/EM/Europe/Asia/LatAm sums --
    if "Market_Cap" in out:
        mc = out["Market_Cap"].copy()
        _agg_specs = [
            ("DM", dm_countries),
            ("EM", em_countries),
            ("Europe", europe_countries),
            ("Asia", asia_countries),
            ("LatAm", latam_countries),
        ]
        for label, countries in _agg_specs:
            cols = [c for c in countries if c in mc.columns]
            if cols:
                mc[label] = mc[cols].sum(axis=1)
        out["Market_Cap"] = mc

    # -- Ten_Year: divide by 100 first, then weighted averages --
    if "Ten_Year" in out:
        ty = out["Ten_Year"].copy() / 100  # convert from % to decimal

        def _weighted_avg(countries_list: list[str]) -> Optional[pd.Series]:
            valid_cols = [c for c in countries_list if c in ty.columns]
            if not valid_cols:
                return None
            data = ty[valid_cols]
            row_sums = data.sum(axis=1)
            weights = data.div(row_sums, axis=0)
            return (data * weights).sum(axis=1)

        _ty_agg_specs = [
            ("DM", dm_countries),
            ("EM", em_countries),
            ("Asia", asia_countries),
            ("Europe", europe_countries),
            ("LatAm", latam_countries),
            ("World", world_indices),
        ]
        for label, countries in _ty_agg_specs:
            result = _weighted_avg(countries)
            if result is not None:
                ty[label] = result

        out["Ten_Year"] = ty

    return out


# ---------------------------------------------------------------------------
# New momentum factors
# ---------------------------------------------------------------------------

def add_price_momentum(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Momentum_12_1 = P(t-21)/P(t-252) - 1 ; Momentum_6_1 = P(t-21)/P(t-126) - 1."""
    if "Price" not in dfs:
        _warn_missing("add_price_momentum", ["Price"])
        return dfs

    px = dfs["Price"]
    out = dict(dfs)
    out["Momentum_12_1"] = px.shift(21) / px.shift(252) - 1.0
    out["Momentum_6_1"] = px.shift(21) / px.shift(126) - 1.0
    return out


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_processing(
    dfs: dict[str, pd.DataFrame],
    classification: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Chain all derived-metric steps in legacy order, with momentum appended last.

    Steps (matching ProcessData.transform_process_data ordering):
        2. Regional aggregates
        3. Consensus growth
        4. Yields
        5. Spreads
        6. Reconstructions
        7. Margins
        8. Rolling stats
        9. Leverage
        +  Price momentum (new)

    Parameters
    ----------
    dfs:
        Raw or ingestion-cleaned metric dict.  Not mutated.
    classification:
        Country classification DataFrame (index=country, cols=Segment/Region).

    Returns
    -------
    dict[str, pd.DataFrame]
        Enriched metric dict.
    """
    result = add_regional_aggregates(dfs, classification)
    result = add_consensus_growth(result)
    result = add_yields(result)
    result = add_spreads(result)
    result = add_reconstructions(result)
    result = add_margins(result)
    result = add_rolling_stats(result)
    result = add_leverage(result)
    result = add_price_momentum(result)
    return result
