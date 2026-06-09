"""
Information Coefficient (IC) analysis.

Matches legacy math in test_normalized_scores.py lines ~193-301 exactly:

Date grid
---------
  selected_dates = sorted(scores.index[::-periodicity])
  (reverse-sampling then sort — identical to legacy calculate_period_returns /
   calculate_signal; guarantees consecutive grid dates are exactly `periodicity`
   rows apart in the original index.)

Signal
------
  absolute: signal at grid date i  = score[grid[i]]
  relative: signal at grid date i  = score[grid[i]] - score[grid[i-1]]
            (diff on the sampled frame, first row dropped — legacy line ~190)

Forward return
--------------
  fwd_ret[i] = price[grid[i+1]] / price[grid[i]] - 1  per country
  (IC pairs signal at grid[i] with return from grid[i] to grid[i+1].)

Spearman rank correlation; minimum 3 common non-NaN countries, else skip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def information_coefficient(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    periodicity: int,
    method: str,
) -> pd.DataFrame:
    """
    Compute Spearman IC between signals and one-period-ahead returns on a
    reverse-sampled date grid.

    Parameters
    ----------
    scores : pd.DataFrame
        Factor scores, rows = dates, columns = countries.
    prices : pd.DataFrame
        Price levels, rows = dates, columns = countries.
    periodicity : int
        Grid step in index rows (e.g. 21 for monthly).
    method : str
        'absolute'  → signal = score level at grid date.
        'relative'  → signal = score[t] - score[t - periodicity].

    Returns
    -------
    pd.DataFrame
        Index = signal dates (grid[i] for each valid pair).
        Columns: "IC" (Spearman correlation), "n_countries" (int).
    """
    if method not in ("absolute", "relative"):
        raise ValueError(f"method must be 'absolute' or 'relative', got {method!r}")

    # --- build grid (legacy: all_dates[::-periodicity] then sorted) ---
    all_dates = scores.index
    grid = sorted(all_dates[::-periodicity])

    # intersect with prices to avoid KeyErrors
    price_dates = set(prices.index)
    grid = [d for d in grid if d in price_dates]

    if len(grid) < 2:
        return pd.DataFrame(columns=["IC", "n_countries"])

    # --- sample scores at grid dates ---
    scores_sampled = scores.reindex(grid)

    # --- build signal series ---
    if method == "absolute":
        signal_df = scores_sampled
    else:
        # relative: diff on sampled frame, drop first row (legacy line ~190)
        signal_df = scores_sampled.diff().iloc[1:]
        # grid is now one shorter on the left
        grid = grid[1:]

    # prices sampled at grid for forward-return computation
    prices_sampled = prices.reindex(grid)

    # --- iterate pairs (grid[i], grid[i+1]) ---
    records: list[dict] = []
    for i in range(len(grid) - 1):
        t0 = grid[i]
        t1 = grid[i + 1]

        signal = signal_df.loc[t0]
        p0 = prices_sampled.loc[t0]
        p1 = prices_sampled.loc[t1]

        # Guard nonpositive start prices: forward return is undefined → NaN
        fwd_ret = p1 / p0.where(p0 > 0) - 1.0

        # common non-NaN countries
        valid = signal.dropna().index.intersection(fwd_ret.dropna().index)
        if len(valid) < 3:
            continue

        ic_val, _ = spearmanr(signal[valid].values, fwd_ret[valid].values)
        records.append({"date": t0, "IC": float(ic_val), "n_countries": len(valid)})

    if not records:
        return pd.DataFrame(columns=["IC", "n_countries"])

    out = pd.DataFrame(records).set_index("date")
    out.index.name = None
    return out


def ic_stats(ic: pd.Series) -> dict:
    """
    Summary statistics for an IC time series.

    Matches legacy calculate_ic_statistics math (lines ~256-269):
      t_stat  = mean / (std / sqrt(n))   ≡ mean / std * sqrt(n)
      icir    = mean / std
      hit_rate = (ic > 0).mean()

    Parameters
    ----------
    ic : pd.Series
        Series of IC values (one per period).

    Returns
    -------
    dict with keys:
        mean_ic, median_ic, std_ic, t_stat, icir, hit_rate
    """
    clean = ic.dropna()
    n = len(clean)

    if n == 0:
        return {
            "mean_ic": np.nan,
            "median_ic": np.nan,
            "std_ic": np.nan,
            "t_stat": np.nan,
            "icir": np.nan,
            "hit_rate": np.nan,
        }

    mean_ic = float(clean.mean())
    median_ic = float(clean.median())
    std_ic = float(clean.std())

    # legacy: divide by zero → 0, but NaN is safer; test verifies exact formula
    if std_ic == 0.0:
        t_stat = np.nan
        icir = np.nan
    else:
        t_stat = mean_ic / std_ic * np.sqrt(n)   # == mean / (std/sqrt(n))
        icir = mean_ic / std_ic

    hit_rate = float((clean > 0).mean()) if n > 0 else np.nan

    return {
        "mean_ic": mean_ic,
        "median_ic": median_ic,
        "std_ic": std_ic,
        "t_stat": t_stat,
        "icir": icir,
        "hit_rate": hit_rate,
    }
