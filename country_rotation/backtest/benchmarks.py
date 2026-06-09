"""Passive benchmark helpers for strategy evaluation.

Provides the honest passive null against which a rotation strategy should be
compared.  No rebalancing — positions are initiated once and held.
"""
from __future__ import annotations

import pandas as pd


def equal_weight_buy_hold(
    prices: pd.DataFrame,
    start=None,
) -> pd.Series:
    """Equal-weight, buy-once-hold equity curve of the universe.

    The honest passive null for a rotation strategy.  Each column is rebased
    to 1.0 at the first available date (>= start if given), the mean across
    columns is taken at every date, and the result is renormalised to start at
    1.0.

    Columns that contain *any* NaN over the selected window are dropped before
    computation so the curve has no gaps.

    Parameters
    ----------
    prices : pd.DataFrame
        Price levels (dates × countries).  Index must be sorted.
    start : label, optional
        If given, the series is sliced to ``prices.loc[start:]`` before
        processing.  The first row of the sliced frame becomes the base date.

    Returns
    -------
    pd.Series
        Equity curve indexed like the (sliced) prices DataFrame, starting at
        exactly 1.0 with no NaN values.

    Raises
    ------
    ValueError
        If no column is NaN-free over the selected window.
    """
    if start is not None:
        prices = prices.loc[start:]

    # Drop any column with at least one NaN in this window
    clean = prices.dropna(axis=1)
    if clean.shape[1] == 0:
        raise ValueError(
            "equal_weight_buy_hold: no columns survive dropna — every country "
            "has at least one NaN over the selected window."
        )

    # Rebase each column to 1.0 at the first row
    rebased = clean / clean.iloc[0]

    # Equal-weight mean across columns (no rebalancing — weights drift with prices)
    equity = rebased.mean(axis=1)

    # Renormalise to exactly 1.0 at first date (floating-point safety)
    equity = equity / equity.iloc[0]

    return equity
