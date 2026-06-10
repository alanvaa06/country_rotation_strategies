"""OOS-honest walk-forward factor screening (Task B3, decision D6).

Screens candidate factors on anchored (expanding) train folds, gates them on
mean IC sign, sign consistency across folds, and a Benjamini-Hochberg
false-discovery-rate test, labels survivors with a raw |t| < 3 "weak" flag
(Harvey-Liu-Zhu), and reserves a terminal lockbox slice that the screening
NEVER reads.  :func:`verify_on_lockbox` is the single, one-shot evaluation
of kept factors on that lockbox.

Statistical power rationale
---------------------------
The per-factor p-value comes from a t-test on the **full per-period IC series**
over the entire screening window (df = n_periods - 1 instead of n_folds - 1 = 4).
With monthly periodicity and a multi-year screening window this gives df = 50-150+
rather than 4, yielding dramatically better statistical power (the "Grinold-Kahn"
approach: treat each IC observation as approximately i.i.d. under the null).

Overlapping-window caveat: the per-period ICs are computed on the full screening
window (not chunked by fold), so they are NOT anchored-expanding — consecutive IC
observations use overlapping data.  This makes the t-test *optimistic* relative to
a block-bootstrap, but far *less* optimistic than the 5-fold version it replaces
(where 5 correlated fold means masqueraded as 5 independent observations).  The
remaining regime-robustness guard (sign consistency across anchored folds) mitigates
the optimism at the cost of some power on factors with time-varying signal.

The fold columns are still computed and reported in ``fold_ic`` for the
sign-consistency regime-robustness gate; they are NOT used for the p-value.

Pure numpy/pandas/scipy; no IO; deterministic. All results are frozen
dataclasses.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from country_rotation.backtest import ic as ic_module

#: Harvey, Liu & Zhu (2016) hurdle: kept factors with raw |t| below this
#: are labelled "weak" rather than trusted.
HLZ_T_THRESHOLD = 3.0

#: Minimum number of per-period IC observations required to compute a
#: meaningful t-stat; series shorter than this receive p=1.0.
MIN_IC_OBS = 8


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FactorScreenResult:
    """Output of :func:`screen_factors` (optionally enriched by
    :func:`verify_on_lockbox`).

    Attributes
    ----------
    fold_ic : pd.DataFrame
        Rows = factor names, columns = ``fold_1 .. fold_k``; each cell is
        the mean IC of that factor on that (train-only) anchored fold.
        Used for the sign-consistency regime-robustness gate only; NOT used
        to compute p-values (see ``series_t``).
    kept : tuple
        Factors passing all gates (positive mean, sign consistency, BH-FDR).
    weak : tuple
        Subset of ``kept`` whose IC-series t-stat has ``|t| <`` the HLZ
        threshold of 3.0 — statistically fragile survivors.
    dropped : tuple
        Factors failing at least one gate (``kept + dropped`` partitions
        the input).
    bh_qvalues : dict
        Factor name -> Benjamini-Hochberg adjusted q-value.
    series_t : dict
        Factor name -> t-statistic from the per-period IC series
        (df = n_ic_obs - 1).  This is the stat used for BH p-values and the
        HLZ weak label; it has far more power than the old fold-mean t-test.
    n_ic_obs : dict
        Factor name -> number of non-NaN IC observations in the screening
        window used to compute ``series_t``.
    lockbox_ic : dict | None
        ``None`` after screening; filled (kept factors only) by
        :func:`verify_on_lockbox`.
    """

    fold_ic: pd.DataFrame
    kept: tuple
    weak: tuple
    dropped: tuple
    bh_qvalues: dict
    series_t: dict
    n_ic_obs: dict
    lockbox_ic: dict | None


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvalues: dict, q: float) -> dict:
    """Benjamini-Hochberg adjusted q-values (standard step-up procedure).

    Sort p-values ascending, compute ``raw_i = p_i * m / rank_i``, enforce
    monotonicity from the largest p downwards (``q_i = min(raw_i, q_{i+1})``)
    and cap at 1.  A factor passes FDR control at level ``alpha`` iff its
    adjusted q-value is ``<= alpha``.

    Parameters
    ----------
    pvalues :
        Factor name -> raw p-value.  NaN p-values are treated as 1.0.
    q :
        Target FDR level of the caller.  The adjusted q-values themselves
        are threshold-independent; the parameter documents the decision
        level and keeps the call-site explicit.

    Returns
    -------
    dict
        Factor name -> adjusted q-value (same keys as ``pvalues``).
    """
    del q  # adjusted q-values do not depend on the decision threshold
    if not pvalues:
        return {}

    names = list(pvalues)
    p = np.array([pvalues[name] for name in names], dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    m = len(p)

    order = np.argsort(p, kind="stable")           # ascending p
    raw = p[order] * m / np.arange(1, m + 1)       # p_i * m / rank_i
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]   # monotone from top
    adjusted = np.minimum(adjusted, 1.0)

    return {names[idx]: float(adjusted[pos]) for pos, idx in enumerate(order)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_dates(
    factor_scores: dict, prices: pd.DataFrame, lockbox_frac: float
) -> tuple:
    """Common sorted dates split into (screen_dates, lockbox_dates).

    Dates = intersection of every factor's index and the price index,
    sorted ascending.  The lockbox is the LAST ``floor(lockbox_frac * n)``
    dates; screening must only ever consume ``screen_dates``.
    """
    common = None
    for df in factor_scores.values():
        common = df.index if common is None else common.intersection(df.index)
    common = common.intersection(prices.index)
    dates = common.sort_values()

    n_lockbox = int(lockbox_frac * len(dates))
    split = len(dates) - n_lockbox
    return dates[:split], dates[split:]


def _mean_ic(
    factor: pd.DataFrame, prices: pd.DataFrame, dates: pd.Index, periodicity: int
) -> float:
    """Mean Spearman IC of one factor over one date slice (NaN if empty)."""
    if len(dates) == 0:
        return float("nan")
    out = ic_module.information_coefficient(
        factor.loc[dates], prices.loc[dates], periodicity, method="absolute"
    )
    series = out["IC"].dropna() if "IC" in out.columns else pd.Series(dtype=float)
    return float(series.mean()) if len(series) else float("nan")


def _series_pvalue(ic_series: pd.Series) -> tuple:
    """One-sided t-test on the full per-period IC series (df = n-1).

    Returns ``(t_stat, p_value, n_obs)``.

    Power rationale
    ~~~~~~~~~~~~~~~
    Using n_periods - 1 degrees of freedom (where n_periods is the number of
    non-NaN IC observations over the entire screening window) gives df = 50-150+
    for typical multi-year screens at monthly periodicity — compared to df = 4
    from the old fold-mean t-test.  This is the Grinold-Kahn approach.

    Degenerate cases:
      - n < MIN_IC_OBS (8): too few observations → p = 1.0.
      - std == 0 with mean <= 0: no evidence → p = 1.0.
      - std == 0 with mean > 0: unanimous positive signal → p = 0.0.
      - std is non-finite: p = 1.0.
    """
    clean = ic_series.dropna()
    n = len(clean)

    if n < MIN_IC_OBS:
        return float("nan"), 1.0, n

    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=1))

    if not np.isfinite(std_val):
        return float("nan"), 1.0, n

    if std_val == 0.0:
        if mean_val > 0.0:
            return float("inf"), 0.0, n
        return float("nan"), 1.0, n

    t = mean_val / (std_val / np.sqrt(n))
    p = float(1.0 - stats.t.cdf(t, df=n - 1))
    return float(t), p, n


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def screen_factors(
    factor_scores: dict,
    prices: pd.DataFrame,
    periodicity: int = 21,
    n_folds: int = 5,
    lockbox_frac: float = 0.2,
    min_sign_consistency: float = 0.7,
    fdr_q: float = 0.10,
    exploratory: set | None = None,
    min_mean_ic: float = 0.0,
) -> FactorScreenResult:
    """Screen factors on anchored train folds with a never-touched lockbox.

    Mechanics
    ---------
    1. ``dates`` = common index of all factors intersected with
       ``prices.index`` (sorted).  The last ``floor(lockbox_frac * n)``
       dates form the lockbox and are EXCLUDED from every step below.
    2. **Per-period IC series** (power gate): for each factor, compute the
       full Spearman IC series over all of ``screen_dates`` using
       :func:`country_rotation.backtest.ic.information_coefficient`
       (``method='absolute'``).  The t-statistic
       ``t = mean(IC) / (std(IC) / sqrt(n))`` has df = n-1 where n is the
       number of non-NaN IC observations.  This is the Grinold-Kahn
       statistic — with monthly periodicity over a 3-year+ screening window
       df ≈ 30-150, vastly more powerful than the old df=4 fold-mean test.
    3. **Anchored fold means** (regime-robustness gate): fold ``k`` (1-based)
       trains on ``screen_dates[: int(len * k / n_folds)]``.  Per factor and
       fold, the mean Spearman IC is computed.  These fold means are used
       ONLY for the sign-consistency gate (step 4b) and are reported in
       ``fold_ic`` for diagnostics.  They do NOT contribute to the p-value.
    4. Gates (all must pass to be kept):
       (a) mean of the per-period IC series > ``min_mean_ic`` (default 0);
       (b) sign consistency ``frac(fold mean > 0) >= min_sign_consistency``
           (NaN folds count as failures) — regime-robustness guard;
       (c) BH-FDR: adjusted q <= ``fdr_q`` (``fdr_q / 2`` for factors in
           ``exploratory`` — data-mined candidates get a stricter bar).
    5. Kept factors with raw IC-series ``|t| <`` :data:`HLZ_T_THRESHOLD`
       are additionally labelled ``weak``.

    Statistical notes
    ~~~~~~~~~~~~~~~~~
    * The per-period IC series uses the full screening window (not expanding
      folds), so consecutive observations are based on overlapping data.
      The t-test assumes approximate i.i.d. ICs; for monthly periodicity with
      autocorrelated IC this is optimistic, but substantially less so than
      treating 5 correlated fold means as 5 independent observations.
    * The sign-consistency gate (folds still computed as before) provides a
      regime-robustness check that is insensitive to the IC-series
      autocorrelation issue.
    * The BH-FDR adjustment corrects for the number of factors tested.

    Parameters
    ----------
    factor_scores :
        Factor name -> per-country score DataFrame (dates x countries).
    prices :
        Price levels (dates x countries); extra columns are ignored by the
        IC's per-date country intersection.
    periodicity :
        IC grid step in index rows (21 = monthly).
    n_folds :
        Number of anchored expanding folds (used ONLY for sign-consistency
        regime-robustness gate; not used for p-value computation).
    lockbox_frac :
        Fraction of the common dates reserved (at the end) as the lockbox.
    min_sign_consistency :
        Minimum fraction of folds with a positive mean IC.
    fdr_q :
        BH false-discovery-rate level.
    exploratory :
        Factor names held to the stricter ``fdr_q / 2`` bar.
    min_mean_ic :
        Minimum required mean IC (from the per-period IC series) for gate (a).
        Default 0.0 (any positive mean passes).

    Returns
    -------
    FactorScreenResult
        With ``lockbox_ic=None`` — only :func:`verify_on_lockbox` may open
        the lockbox.  ``series_t`` and ``n_ic_obs`` are populated for every
        factor.
    """
    if not factor_scores:
        raise ValueError("factor_scores must contain at least one factor")
    exploratory = set() if exploratory is None else set(exploratory)

    screen_dates, _ = _split_dates(factor_scores, prices, lockbox_frac)
    n_screen = len(screen_dates)

    # ------------------------------------------------------------------
    # Step 1: per-period IC series over the FULL screening window
    # ------------------------------------------------------------------
    ic_series_map: dict = {}
    for name, factor in factor_scores.items():
        ic_df = ic_module.information_coefficient(
            factor.loc[screen_dates], prices.loc[screen_dates], periodicity, "absolute"
        )
        ic_series_map[name] = ic_df["IC"].dropna() if "IC" in ic_df.columns else pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Step 2: anchored fold means (regime-robustness gate only)
    # ------------------------------------------------------------------
    fold_ends = [int(n_screen * k / n_folds) for k in range(1, n_folds + 1)]
    fold_cols = [f"fold_{k}" for k in range(1, n_folds + 1)]
    fold_rows = {
        name: [
            _mean_ic(factor, prices, screen_dates[:end], periodicity)
            for end in fold_ends
        ]
        for name, factor in factor_scores.items()
    }
    fold_ic = pd.DataFrame.from_dict(fold_rows, orient="index", columns=fold_cols)
    fold_ic = fold_ic.astype(float)

    # ------------------------------------------------------------------
    # Step 3: t-stats and p-values from the per-period IC series
    # ------------------------------------------------------------------
    series_t: dict = {}
    n_ic_obs: dict = {}
    pvalues: dict = {}
    for name in factor_scores:
        t, p, n_obs = _series_pvalue(ic_series_map[name])
        series_t[name] = t
        n_ic_obs[name] = n_obs
        pvalues[name] = p

    bh_qvalues = benjamini_hochberg(pvalues, fdr_q)

    # ------------------------------------------------------------------
    # Step 4: apply gates
    # ------------------------------------------------------------------
    kept_list: list = []
    weak_list: list = []
    for name in factor_scores:
        ic_ser = ic_series_map[name]
        fold_means = fold_ic.loc[name].to_numpy(dtype=float)
        valid_folds = fold_means[np.isfinite(fold_means)]

        # Gate (a): positive mean IC from the per-period series
        series_mean = float(ic_ser.mean()) if len(ic_ser) > 0 else float("nan")
        mean_positive = np.isfinite(series_mean) and series_mean > min_mean_ic

        # Gate (b): sign consistency across anchored folds (regime robustness)
        sign_consistency = float((valid_folds > 0.0).sum()) / n_folds  # NaN folds fail

        # Gate (c): BH-FDR
        q_bar = fdr_q / 2.0 if name in exploratory else fdr_q
        fdr_pass = bh_qvalues[name] <= q_bar

        if mean_positive and sign_consistency >= min_sign_consistency and fdr_pass:
            kept_list.append(name)
            if abs(series_t[name]) < HLZ_T_THRESHOLD:
                weak_list.append(name)

    kept = tuple(kept_list)
    dropped = tuple(name for name in factor_scores if name not in set(kept))

    return FactorScreenResult(
        fold_ic=fold_ic,
        kept=kept,
        weak=tuple(weak_list),
        dropped=dropped,
        bh_qvalues=bh_qvalues,
        series_t=series_t,
        n_ic_obs=n_ic_obs,
        lockbox_ic=None,
    )


# ---------------------------------------------------------------------------
# Lockbox verification
# ---------------------------------------------------------------------------

def verify_on_lockbox(
    result: FactorScreenResult,
    factor_scores: dict,
    prices: pd.DataFrame,
    periodicity: int = 21,
    lockbox_frac: float = 0.2,
) -> FactorScreenResult:
    """One-shot lockbox evaluation of the KEPT factors only.

    Recomputes the mean IC on the lockbox slice (the last
    ``floor(lockbox_frac * n)`` common dates — same boundary arithmetic as
    :func:`screen_factors`) for each kept factor, and returns a copy of
    ``result`` with ``lockbox_ic`` filled.  The input ``result`` is frozen
    and left untouched.  Run this exactly once and report lockbox ICs
    side-by-side with the train fold ICs; re-running after changing the
    screen is lockbox burn.

    Returns
    -------
    FactorScreenResult
        ``dataclasses.replace(result, lockbox_ic={factor: mean lockbox IC})``.
    """
    _, lockbox_dates = _split_dates(factor_scores, prices, lockbox_frac)

    lockbox_ic = {
        name: _mean_ic(factor_scores[name], prices, lockbox_dates, periodicity)
        for name in result.kept
    }
    return dataclasses.replace(result, lockbox_ic=lockbox_ic)
