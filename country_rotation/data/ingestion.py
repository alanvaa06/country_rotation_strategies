"""Data ingestion layer: read, clean, and align raw market data files.

Public API
----------
read_inputs          -- load all xlsx files from a folder into a dict
load_classification  -- load country-segment classification from Excel
remove_weekends      -- filter Saturday/Sunday rows from every DataFrame
slice_by_date        -- keep rows >= target_date
drop_countries       -- remove listed column names (errors='ignore')
apply_publication_lag -- shift metrics forward by their publication lag
fill_gaps            -- ffill-only gap filling (NO bfill; no future leakage)
"""
from __future__ import annotations

import os
import warnings
from typing import Union

import pandas as pd


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_inputs(folder: str, skiprows: int = 2) -> dict[str, pd.DataFrame]:
    """Read all Excel files in *folder* and return a metric-keyed dict.

    Parameters
    ----------
    folder:
        Directory containing .xlsx / .xls files.
    skiprows:
        Number of header rows to skip (legacy default = 2).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are filenames without extension; values are DataFrames with a
        DatetimeIndex (index_col=0, parse_dates=True).

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    """
    result: dict[str, pd.DataFrame] = {}

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Inputs folder '{folder}' does not exist.")

    for filename in os.listdir(folder):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(folder, filename)
            try:
                df = pd.read_excel(
                    file_path,
                    skiprows=skiprows,
                    index_col=0,
                    parse_dates=True,
                )
                key = filename.replace(".xlsx", "").replace(".xls", "")
                result[key] = df
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Could not read {filename}. Error: {exc}", stacklevel=2
                )

    return result


def load_classification(path: str) -> pd.DataFrame:
    """Load the *regiones* sheet of the classification workbook.

    Parameters
    ----------
    path:
        Absolute path to the Classification.xlsx file.

    Returns
    -------
    pd.DataFrame
        Index = country names; columns include Segment, Region, Type.
    """
    return pd.read_excel(path, index_col=0, sheet_name="regiones")


# ---------------------------------------------------------------------------
# Pure transforms (no I/O)
# ---------------------------------------------------------------------------

def remove_weekends(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return a new dict with Saturday/Sunday rows removed from every DataFrame.

    Parameters
    ----------
    dfs:
        Input metric dict.

    Returns
    -------
    dict[str, pd.DataFrame]
        Same keys; each DataFrame filtered to weekdays only (dayofweek < 5).
    """
    out: dict[str, pd.DataFrame] = {}

    for key, df in dfs.items():
        if df.empty:
            out[key] = df
            continue

        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        mask = df.index.dayofweek < 5
        out[key] = df[mask].copy()

    return out


def slice_by_date(
    dfs: dict[str, pd.DataFrame],
    target_date: str,
) -> dict[str, pd.DataFrame]:
    """Keep only rows with index >= *target_date*.

    Parameters
    ----------
    dfs:
        Input metric dict.
    target_date:
        Inclusive start date in 'YYYY-MM-DD' format.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    target_dt = pd.to_datetime(target_date)
    out: dict[str, pd.DataFrame] = {}

    for key, df in dfs.items():
        if df.empty:
            out[key] = df
            continue

        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        mask = df.index >= target_dt
        out[key] = df[mask].copy()

    return out


def drop_countries(
    dfs: dict[str, pd.DataFrame],
    columns_to_drop: tuple,
) -> dict[str, pd.DataFrame]:
    """Drop listed columns from every DataFrame (silently ignores missing names).

    Parameters
    ----------
    dfs:
        Input metric dict.
    columns_to_drop:
        Tuple of column names to remove.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    out: dict[str, pd.DataFrame] = {}

    for key, df in dfs.items():
        out[key] = df.drop(columns=list(columns_to_drop), errors="ignore")

    return out


def apply_publication_lag(
    dfs: dict[str, pd.DataFrame],
    lag_days: dict[str, int],
) -> dict[str, pd.DataFrame]:
    """Shift metrics forward by their publication lag (introduces NaNs at head).

    Metrics not in *lag_days* are returned unchanged.  Metrics with lag == 0
    are also returned unchanged (no-op).

    Parameters
    ----------
    dfs:
        Input metric dict.
    lag_days:
        Mapping of metric name -> integer lag in trading periods.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    out = dict(dfs)

    for metric, lag in lag_days.items():
        if metric in out and lag > 0:
            out[metric] = out[metric].shift(lag)

    return out


def fill_gaps(
    dfs: dict[str, pd.DataFrame],
    metrics: tuple,
    limit: int,
) -> dict[str, pd.DataFrame]:
    """Forward-fill gaps in listed metrics — strictly no backward fill.

    Uses ``ffill(limit=limit)`` only.  This guarantees that no future
    observation can propagate into the past (unlike the legacy ``bfill``
    call at ProcessData.py ~line 350).

    Parameters
    ----------
    dfs:
        Input metric dict.
    metrics:
        Tuple of metric names to gap-fill.
    limit:
        Maximum consecutive NaN periods to fill.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    out = dict(dfs)

    for metric in metrics:
        if metric in out:
            out[metric] = out[metric].ffill(limit=limit)

    return out
