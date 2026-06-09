"""OOS-honest walk-forward factor screening (Task B3, decision D6).

Screens candidate factors on anchored (expanding) train folds, gates them on
mean IC sign, sign consistency across folds, and a Benjamini-Hochberg
false-discovery-rate test, labels survivors with a raw |t| < 3 "weak" flag
(Harvey-Liu-Zhu), and reserves a terminal lockbox slice that the screening
NEVER reads.  :func:`verify_on_lockbox` is the single, one-shot evaluation
of kept factors on that lockbox.

Statistical honesty note
------------------------
The per-factor p-value is a one-sided t-test on the *fold-mean* ICs.
Anchored expanding folds overlap (fold k's window contains fold k-1's), so
the fold means are NOT independent observations and the t-test treats them
as if they were — the effective sample size is smaller than ``n_folds`` and
the t-statistic is optimistic.  This is a known, deliberate approximation:
the sign-consistency and positive-mean gates are blunt instruments that
mitigate it, and the BH-FDR adjustment plus the HLZ |t| >= 3 bar for the
"non-weak" label add further conservatism.  Treat ``bh_qvalues`` as a
screening heuristic, not a calibrated error rate.

Pure numpy/pandas/scipy; no IO; deterministic. All results are frozen
dataclasses.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from country_rotation.backtest import ic

#: Harvey, Liu & Zhu (2016) hurdle: kept factors with raw |t| below this
#: are labelled "weak" rather than trusted.
HLZ_T_THRESHOLD = 3.0


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
    kept : tuple
        Factors passing all gates (positive mean, sign consistency, BH-FDR).
    weak : tuple
        Subset of ``kept`` whose raw fold-mean t-stat has ``|t| <`` the HLZ
        threshold of 3.0 — statistically fragile survivors.
    dropped : tuple
        Factors failing at least one gate (``kept + dropped`` partitions
        the input).
    bh_qvalues : dict
        Factor name -> Benjamini-Hochberg adjusted q-value.
    lockbox_ic : dict | None
        ``None`` after screening; filled (kept factors only) by
        :func:`verify_on_lockbox`.
    """

    fold_ic: pd.DataFrame
    kept: tuple
    weak: tuple
    dropped: tuple
    bh_qvalues: dict
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
    out = ic.information_coefficient(
        factor.loc[dates], prices.loc[dates], periodicity, method="absolute"
    )
    series = out["IC"].dropna() if "IC" in out.columns else pd.Series(dtype=float)
    return float(series.mean()) if len(series) else float("nan")


def _fold_pvalue(fold_means: np.ndarray) -> tuple:
    """One-sided t-test that the mean of fold-mean ICs exceeds zero.

    Returns ``(t_stat, p_value)``.  NaN fold means are dropped.  Degenerate
    cases (< 2 valid folds, non-finite std) return ``(nan, 1.0)``.  A zero
    std splits by sign: a strictly positive constant fold-mean vector is
    unanimous evidence for the factor (``t=+inf, p=0.0`` — e.g. a signal
    with Spearman IC exactly 1 every period), while a non-positive constant
    carries no supporting evidence (``p=1.0``).
    See the module docstring: overlapping anchored folds make this an
    optimistic approximation, mitigated by the other gates.
    """
    valid = fold_means[np.isfinite(fold_means)]
    if len(valid) < 2:
        return float("nan"), 1.0
    sd = valid.std(ddof=1)
    if not np.isfinite(sd):
        return float("nan"), 1.0
    if sd == 0.0:
        if valid.mean() > 0.0:
            return float("inf"), 0.0
        return float("nan"), 1.0
    t = float(valid.mean() / (sd / np.sqrt(len(valid))))
    p = float(1.0 - stats.t.cdf(t, df=len(valid) - 1))
    return t, p


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
) -> FactorScreenResult:
    """Screen factors on anchored train folds with a never-touched lockbox.

    Mechanics
    ---------
    1. ``dates`` = common index of all factors intersected with
       ``prices.index`` (sorted).  The last ``floor(lockbox_frac * n)``
       dates form the lockbox and are EXCLUDED from every step below.
    2. Anchored expanding folds over the remaining screen dates: fold ``k``
       (1-based) trains on ``screen_dates[: int(len * k / n_folds)]``.
       Per factor and fold, the mean Spearman IC is computed with
       :func:`country_rotation.backtest.ic.information_coefficient`
       (``method='absolute'``).
    3. Gates (all must pass to be kept):
       (a) mean of fold-mean ICs > 0;
       (b) sign consistency ``frac(fold mean > 0) >= min_sign_consistency``
           (NaN folds count as failures);
       (c) BH-FDR: adjusted q <= ``fdr_q`` (``fdr_q / 2`` for factors in
           ``exploratory`` — data-mined candidates get a stricter bar).
    4. Kept factors with raw ``|t| <`` :data:`HLZ_T_THRESHOLD` are
       additionally labelled ``weak``.

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
        Number of anchored expanding folds.
    lockbox_frac :
        Fraction of the common dates reserved (at the end) as the lockbox.
    min_sign_consistency :
        Minimum fraction of folds with a positive mean IC.
    fdr_q :
        BH false-discovery-rate level.
    exploratory :
        Factor names held to the stricter ``fdr_q / 2`` bar.

    Returns
    -------
    FactorScreenResult
        With ``lockbox_ic=None`` — only :func:`verify_on_lockbox` may open
        the lockbox.
    """
    if not factor_scores:
        raise ValueError("factor_scores must contain at least one factor")
    exploratory = set() if exploratory is None else set(exploratory)

    screen_dates, _ = _split_dates(factor_scores, prices, lockbox_frac)
    n_screen = len(screen_dates)
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

    tstats: dict = {}
    pvalues: dict = {}
    for name in factor_scores:
        t, p = _fold_pvalue(fold_ic.loc[name].to_numpy(dtype=float))
        tstats[name] = t
        pvalues[name] = p

    bh_qvalues = benjamini_hochberg(pvalues, fdr_q)

    kept_list: list = []
    weak_list: list = []
    for name in factor_scores:
        fold_means = fold_ic.loc[name].to_numpy(dtype=float)
        valid = fold_means[np.isfinite(fold_means)]

        mean_positive = len(valid) > 0 and float(valid.mean()) > 0.0
        sign_consistency = float((valid > 0.0).sum()) / n_folds  # NaN folds fail
        q_bar = fdr_q / 2.0 if name in exploratory else fdr_q
        fdr_pass = bh_qvalues[name] <= q_bar

        if mean_positive and sign_consistency >= min_sign_consistency and fdr_pass:
            kept_list.append(name)
            if abs(tstats[name]) < HLZ_T_THRESHOLD:
                weak_list.append(name)

    kept = tuple(kept_list)
    dropped = tuple(name for name in factor_scores if name not in set(kept))

    return FactorScreenResult(
        fold_ic=fold_ic,
        kept=kept,
        weak=tuple(weak_list),
        dropped=dropped,
        bh_qvalues=bh_qvalues,
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
