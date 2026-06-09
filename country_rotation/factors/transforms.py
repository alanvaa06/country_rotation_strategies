"""Factor metric transforms — pure functions ported from legacy FactorTransformer.py.

Each function takes a DataFrame (dates × countries) and returns a DataFrame of the
same shape and index.  No slicing, no side-effects.

Parity source: FactorTransformer.py lines 314-416
  - zscore_percentile  : lines 314-329  (expanding mean/std → shift(1) → norm.cdf)
  - absolute_percentile: lines 331-341  (expanding().rank(pct=True, method='min'))
  - relative_rank      : lines 343-353  (rank(axis=1, pct=True, method='average'))
  - delta_percentile   : lines 355-366  (pct_change(63) → expanding().rank(pct=True, method='average'))
  - direction inversion: lines 392-398  (1.0 - x when direction == -1)

Divergences from task-spec sketch (kept to match legacy):
  - absolute_percentile: method='min'    (spec sketch implied default; legacy is explicit 'min')
  - delta_percentile   : method='average' (spec sketch silent; legacy is explicit 'average')
  - Pure functions do NOT apply .iloc[window:] — they return full aligned DataFrames.
    The legacy class slices only for its own downstream consumption; callers decide what
    rows to use.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def zscore_percentile(df: pd.DataFrame, min_periods: int = 63) -> pd.DataFrame:
    """Expanding z-score converted to a percentile via the standard normal CDF.

    At each row t the z-score uses the mean and standard deviation computed
    over rows 0..t-1 (expanding statistics shifted forward by 1 day to avoid
    look-ahead).  Rows with fewer than *min_periods* prior observations will be
    NaN because the expanding window has not yet accumulated enough data.

    Parity: FactorTransformer.calculate_zscore, lines 325-328.

    Args:
        df: DataFrame with dates as index and countries as columns.
        min_periods: Minimum number of observations required to compute a
            non-NaN statistic (default 63, matching legacy self.window).

    Returns:
        DataFrame of percentile scores in [0, 1] with the same shape as *df*.
    """
    expanding_mean = df.expanding(min_periods=min_periods).mean().shift(1)
    expanding_std = df.expanding(min_periods=min_periods).std().shift(1)
    zscores = (df - expanding_mean) / expanding_std
    percentiles = norm.cdf(zscores)
    return pd.DataFrame(percentiles, index=df.index, columns=df.columns)


def absolute_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """Historical percentile rank of each value within its own expanding history.

    Uses method='min' (fraction of past values ≤ current value), matching
    legacy FactorTransformer.calculate_absolute_percentile line 341.

    Parity: FactorTransformer.calculate_absolute_percentile, line 341.

    Args:
        df: DataFrame with dates as index and countries as columns.

    Returns:
        DataFrame of expanding percentile ranks in (0, 1] with the same shape
        as *df*.
    """
    return df.expanding().rank(pct=True, method="min")


def relative_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank at each date (row-wise).

    At each row, countries are ranked relative to each other using
    method='average', matching legacy FactorTransformer.calculate_relative_ranking
    line 353.

    Parity: FactorTransformer.calculate_relative_ranking, line 353.

    Args:
        df: DataFrame with dates as index and countries as columns.

    Returns:
        DataFrame of cross-sectional percentile ranks in (0, 1] with the same
        shape as *df*.
    """
    return df.rank(axis=1, pct=True, method="average")


def delta_percentile(
    df: pd.DataFrame,
    window: int = 63,
    min_periods: int = 63,
) -> pd.DataFrame:
    """Expanding percentile rank of rolling percent-change (momentum).

    Computes the *window*-period percent change, then ranks each value within
    its own expanding history (method='average').  The first *window* rows will
    be NaN because pct_change has no prior reference.

    Parity: FactorTransformer.calculate_delta_percentile, lines 365-366.

    Args:
        df: DataFrame with dates as index and countries as columns.
        window: Look-back period for percent-change (default 63 business days).
        min_periods: Kept for API symmetry; the expanding rank has no hard
            minimum (pct_change NaNs handle the warm-up naturally).

    Returns:
        DataFrame of expanding percentile ranks in (0, 1] with the same shape
        as *df*.
    """
    pct_change = df.pct_change(periods=window)
    return pct_change.expanding().rank(pct=True, method="average")


# ---------------------------------------------------------------------------
# Direction inversion
# ---------------------------------------------------------------------------

def apply_direction(df: pd.DataFrame, direction: int) -> pd.DataFrame:
    """Invert a transform when the factor is inversely related to returns.

    When *direction* == -1, returns ``1.0 - df`` so that lower raw factor
    values map to higher scores.  Any other value leaves *df* unchanged.

    Parity: FactorTransformer.transform_all, lines 392-398.

    Args:
        df: A transform output DataFrame (values in [0, 1]).
        direction: 1 = higher is better (no inversion);
                  -1 = lower is better (invert).

    Returns:
        DataFrame with or without inversion applied.
    """
    if direction == -1:
        return 1.0 - df
    return df


# ---------------------------------------------------------------------------
# Composite convenience function
# ---------------------------------------------------------------------------

def transform_factor(
    df: pd.DataFrame,
    direction: int,
) -> dict[str, pd.DataFrame]:
    """Apply all four metric transforms to a single factor DataFrame.

    Computes the four standardised metrics and applies the direction flag
    to each of them in a single call, matching the structure produced by
    FactorTransformer.transform_all (lines 388-418).

    Args:
        df: Raw factor DataFrame (dates × countries).
        direction: 1 or -1.  Passed through to :func:`apply_direction`.

    Returns:
        Dictionary with keys ``"zscore"``, ``"absolute_pct"``,
        ``"relative_rank"``, ``"delta_pct"`` mapping to the corresponding
        transformed DataFrames.
    """
    raw: dict[str, pd.DataFrame] = {
        "zscore": zscore_percentile(df),
        "absolute_pct": absolute_percentile(df),
        "relative_rank": relative_rank(df),
        "delta_pct": delta_percentile(df),
    }
    return {key: apply_direction(val, direction) for key, val in raw.items()}
