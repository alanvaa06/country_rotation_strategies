"""Validation statistics for country rotation strategy evaluation (Task B1).

Five pure-function statistics.  Functions ``sharpe_significance``,
``probabilistic_sharpe_ratio``, ``deflated_sharpe_ratio``, and
``bootstrap_sharpe_ci`` operate on equity curves (pd.Series of prices
starting at an arbitrary positive level) — internal logic converts to
simple daily returns via ``pct_change().dropna()``.

``newey_west_tstat`` takes a **return series** (pd.Series) directly,
e.g. strategy-minus-null daily returns (NOT an equity curve).

Functions
---------
sharpe_significance         Lo (2002) IID SE; t = SR_d / SE; NaN-safe.
probabilistic_sharpe_ratio  Bailey & LdP (2012).
deflated_sharpe_ratio       Bailey & LdP (2014) with Euler-Mascheroni correction.
newey_west_tstat            Bartlett kernel, input is a return series.
bootstrap_sharpe_ci         Politis-Romano stationary bootstrap, seeded, wrap-around.

All result types are frozen dataclasses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _returns_from_equity(equity: pd.Series) -> np.ndarray:
    """Compute simple daily returns from an equity curve, dropping any leading NaN."""
    r = equity.pct_change().dropna().to_numpy(dtype=float)
    return r


def _daily_sharpe(returns: np.ndarray) -> float:
    """Daily Sharpe (mean/std, ddof=1). Returns NaN when ill-defined."""
    if len(returns) < 2:
        return math.nan
    std = float(np.std(returns, ddof=1))
    if std == 0.0 or math.isnan(std):
        return math.nan
    return float(np.mean(returns)) / std


# ---------------------------------------------------------------------------
# 1. sharpe_significance — Lo (2002)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SharpeSignificance:
    """Result of Lo (2002) IID Sharpe ratio significance test.

    Attributes
    ----------
    sharpe_daily : float
        Sharpe ratio computed on daily returns (mean/std, ddof=1).
    sharpe_ann : float
        Annualized Sharpe ratio: sharpe_daily * sqrt(252).  NaN when
        sharpe_daily is NaN.
    se : float
        Standard error under the IID assumption:
        SE = sqrt((1 + SR^2 / 2) / n).
    t_stat : float
        t = sharpe_daily / SE.  NaN when SE = 0 or n < 2.
    n_obs : int
        Number of daily returns used.
    """

    sharpe_daily: float
    sharpe_ann: float
    se: float
    t_stat: float
    n_obs: int


def sharpe_significance(equity: pd.Series) -> SharpeSignificance:
    """Compute Lo (2002) IID Sharpe significance for an equity curve.

    Parameters
    ----------
    equity:
        Equity curve (levels, not returns).  Monotone starting level not required.

    Returns
    -------
    SharpeSignificance
        NaN values in ``t_stat`` / ``se`` when inputs are degenerate (n < 2,
        zero variance, single observation).
    """
    returns = _returns_from_equity(equity)
    n = len(returns)

    _sqrt252 = math.sqrt(252.0)

    if n < 2:
        return SharpeSignificance(
            sharpe_daily=math.nan,
            sharpe_ann=math.nan,
            se=math.nan,
            t_stat=math.nan,
            n_obs=n,
        )

    sr_d = _daily_sharpe(returns)

    if math.isnan(sr_d):
        return SharpeSignificance(
            sharpe_daily=sr_d,
            sharpe_ann=math.nan,
            se=math.nan,
            t_stat=math.nan,
            n_obs=n,
        )

    # Lo (2002) eq 8: SE = sqrt((1 + SR^2/2) / n)
    se = math.sqrt((1.0 + sr_d ** 2 / 2.0) / n)

    if se == 0.0:
        t_stat = math.nan
    else:
        t_stat = sr_d / se

    return SharpeSignificance(
        sharpe_daily=sr_d,
        sharpe_ann=sr_d * _sqrt252,
        se=se,
        t_stat=t_stat,
        n_obs=n,
    )


# ---------------------------------------------------------------------------
# 2. probabilistic_sharpe_ratio — Bailey & LdP (2012)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PSRResult:
    """Result of the Probabilistic Sharpe Ratio computation.

    Attributes
    ----------
    psr : float
        PSR = Phi(z) where Phi is the standard-normal CDF.
        In (0, 1).  NaN for degenerate inputs.
    z_stat : float
        Z-score: (SR_obs - SR_benchmark) * sqrt(n-1) / sqrt(variance_term).
    sharpe_daily : float
        Observed daily Sharpe (mean/std, ddof=1).
    n_obs : int
        Number of daily returns.
    skewness : float
        Skewness of daily returns (unbiased, Fisher).
    kurtosis : float
        **Non-excess** kurtosis of daily returns (Normal → 3),
        i.e. scipy.stats.kurtosis(r, fisher=False, bias=False).
    """

    psr: float
    z_stat: float
    sharpe_daily: float
    n_obs: int
    skewness: float
    kurtosis: float  # non-excess (Normal = 3)


def probabilistic_sharpe_ratio(
    equity: pd.Series,
    benchmark_sharpe: float = 0.0,
) -> PSRResult:
    """Compute Bailey & LdP (2012) PSR.

    Canonical formula (ground truth: docs/references/validation_formulas.md):

        PSR(SR*) = Phi( (SR − SR*) · sqrt(n−1)
                        / sqrt(1 − γ3·SR + ((γ4 − 1)/4)·SR²) )

    where γ3 = unbiased skewness, γ4 = **non-excess** kurtosis (Normal → 3),
    i.e. ``scipy.stats.kurtosis(r, fisher=False, bias=False)``.

    Parameters
    ----------
    equity :
        Equity curve.
    benchmark_sharpe :
        Reference Sharpe ratio SR* (daily units, same as SR_obs).

    Returns
    -------
    PSRResult
    """
    returns = _returns_from_equity(equity)
    n = len(returns)

    if n < 4:
        return PSRResult(
            psr=math.nan,
            z_stat=math.nan,
            sharpe_daily=math.nan,
            n_obs=n,
            skewness=math.nan,
            kurtosis=math.nan,
        )

    sr_obs = _daily_sharpe(returns)
    if math.isnan(sr_obs):
        return PSRResult(
            psr=math.nan,
            z_stat=math.nan,
            sharpe_daily=sr_obs,
            n_obs=n,
            skewness=math.nan,
            kurtosis=math.nan,
        )

    skew = float(stats.skew(returns, bias=False))
    # Non-excess kurtosis: fisher=False → Normal returns 3
    g4 = float(stats.kurtosis(returns, fisher=False, bias=False))

    # Canonical variance term: 1 - γ3·SR + ((γ4 - 1)/4)·SR²
    variance_term = 1.0 - skew * sr_obs + ((g4 - 1.0) / 4.0) * sr_obs ** 2

    # Guard against negative variance term (pathological series) — clamp to 1e-12
    if variance_term <= 0.0:
        variance_term = 1e-12

    se_hat = math.sqrt(variance_term / (n - 1))

    if se_hat == 0.0:
        return PSRResult(
            psr=math.nan,
            z_stat=math.nan,
            sharpe_daily=sr_obs,
            n_obs=n,
            skewness=skew,
            kurtosis=g4,
        )

    z_stat = (sr_obs - benchmark_sharpe) / se_hat
    psr = float(stats.norm.cdf(z_stat))

    return PSRResult(
        psr=psr,
        z_stat=z_stat,
        sharpe_daily=sr_obs,
        n_obs=n,
        skewness=skew,
        kurtosis=g4,
    )


# ---------------------------------------------------------------------------
# 3. deflated_sharpe_ratio — Bailey & LdP (2014)
# ---------------------------------------------------------------------------

# Euler-Mascheroni constant
_EULER_MASCHERONI = 0.5772156649015328


@dataclass(frozen=True)
class DSRResult:
    """Result of the Deflated Sharpe Ratio computation.

    Attributes
    ----------
    dsr : float
        DSR = PSR(SR0) where SR0 is the deflated benchmark.  In (0, 1).
    expected_max_sharpe : float
        SR0: expected maximum SR under the null of no skill, estimated from
        the distribution of trial Sharpes using the Euler-Mascheroni
        order-statistic formula (no mean term — null assumes E[SR] = 0).
    n_trials : int
        Number of trial Sharpes.
    """

    dsr: float
    expected_max_sharpe: float
    n_trials: int


def _expected_max_sharpe(trial_sharpes: np.ndarray) -> float:
    """Estimate SR0 (deflated benchmark) from Bailey & LdP (2014).

    Canonical formula (ground truth: docs/references/validation_formulas.md):

        SR0 = sigma · ((1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))

    where sigma = std(trial_sharpes, ddof=1), N = number of trials,
    γ = Euler–Mascheroni constant ≈ 0.5772.

    **No mean term** — the null hypothesis assumes E[SR] = 0.

    Edge cases: N ≤ 1 or sigma ≤ 0 / NaN → return 0.0.

    Parameters
    ----------
    trial_sharpes :
        Array of Sharpe ratios from N independent trials (daily units).

    Returns
    -------
    float
        SR0: deflated benchmark Sharpe.
    """
    N = len(trial_sharpes)
    if N <= 1:
        return 0.0

    sigma = float(np.std(trial_sharpes, ddof=1))

    if sigma <= 0.0 or math.isnan(sigma):
        return 0.0

    gamma = _EULER_MASCHERONI

    # Avoid numeric edge case: 1/N → 0 when N large
    p1 = max(1e-10, min(1.0 - 1e-10, 1.0 - 1.0 / N))
    p2 = max(1e-10, min(1.0 - 1e-10, 1.0 - 1.0 / (N * math.e)))

    z1 = float(stats.norm.ppf(p1))
    z2 = float(stats.norm.ppf(p2))

    sr0 = sigma * ((1.0 - gamma) * z1 + gamma * z2)
    return sr0


def deflated_sharpe_ratio(
    equity: pd.Series,
    trial_sharpes: np.ndarray,
) -> DSRResult:
    """Compute Bailey & LdP (2014) Deflated Sharpe Ratio.

    The DSR adjusts the benchmark Sharpe upward to account for multiple testing
    (selection bias across N trials).  SR0 is estimated from the spread of
    trial Sharpes via the Euler-Mascheroni order-statistic approximation.
    DSR = PSR(SR0).

    Parameters
    ----------
    equity :
        Equity curve of the strategy under evaluation.
    trial_sharpes :
        Array of Sharpe ratios from all tested parameter sets / strategies
        (including the current strategy counts as one trial).

    Returns
    -------
    DSRResult
    """
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    exp_max = _expected_max_sharpe(trial_sharpes)

    psr_result = probabilistic_sharpe_ratio(equity, benchmark_sharpe=exp_max)

    return DSRResult(
        dsr=psr_result.psr,
        expected_max_sharpe=exp_max,
        n_trials=len(trial_sharpes),
    )


# ---------------------------------------------------------------------------
# 4. newey_west_tstat — Bartlett kernel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NWResult:
    """Result of a Newey-West HAC t-test on the mean of a return series.

    Attributes
    ----------
    t_stat : float
        t = mean(x) / sqrt(NW_variance / n).
    nw_variance : float
        HAC variance estimate of the series mean.
    mean_return : float
        Mean of the input return series.
    lags : int
        Number of lags used in the Bartlett kernel.
    n_obs : int
        Number of observations.
    """

    t_stat: float
    nw_variance: float
    mean_return: float
    lags: int
    n_obs: int


def newey_west_tstat(
    series: pd.Series,
    lags: Optional[int] = None,
) -> NWResult:
    """Compute a Newey-West HAC t-statistic for the mean of a return series.

    Input is a **return series** (pd.Series), e.g. strategy-minus-null daily
    returns — NOT an equity curve.  Uses Bartlett kernel.

    Default lag = floor(4 · (n/100)^(2/9)), minimum 1 when auto-selected.

    Parameters
    ----------
    series :
        Return series (pd.Series of daily returns).  NaNs are dropped.
    lags :
        Override automatic lag selection (must be >= 0).

    Returns
    -------
    NWResult
    """
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)

    if n < 2:
        return NWResult(
            t_stat=math.nan,
            nw_variance=math.nan,
            mean_return=math.nan,
            lags=0,
            n_obs=n,
        )

    mu = float(np.mean(x))
    demeaned = x - mu

    if lags is None:
        lag_count = max(1, int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    else:
        lag_count = max(0, int(lags))

    # Bartlett-weighted autocovariance sum
    # gamma_0 = (1/n) * sum(u_t^2)
    gamma0 = float(np.mean(demeaned ** 2))
    nw_var = gamma0  # start with lag-0 term

    for lag in range(1, lag_count + 1):
        w = 1.0 - lag / (lag_count + 1.0)  # Bartlett weight
        gamma_lag = float(np.mean(demeaned[lag:] * demeaned[:-lag]))
        nw_var += 2.0 * w * gamma_lag

    # Variance of sample mean = S_NW / n
    var_mean = nw_var / n

    if var_mean <= 0.0 or math.isnan(var_mean):
        return NWResult(
            t_stat=math.nan,
            nw_variance=nw_var,
            mean_return=mu,
            lags=lag_count,
            n_obs=n,
        )

    t_stat = mu / math.sqrt(var_mean)

    return NWResult(
        t_stat=t_stat,
        nw_variance=nw_var,
        mean_return=mu,
        lags=lag_count,
        n_obs=n,
    )


# ---------------------------------------------------------------------------
# 5. bootstrap_sharpe_ci — Politis-Romano stationary bootstrap
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapCI:
    """Result of a stationary bootstrap Sharpe ratio confidence interval.

    Attributes
    ----------
    ci_low : float
        Lower bound of the bootstrap CI (default 2.5th percentile).
        Sharpe is in daily units (mean/std, ddof=1).
    ci_high : float
        Upper bound of the bootstrap CI (default 97.5th percentile).
        Sharpe is in daily units (mean/std, ddof=1).
    sharpe_point : float
        Point estimate of the Sharpe ratio (mean/std, ddof=1, daily units).
    n_boot : int
        Number of bootstrap replications used.
    """

    ci_low: float
    ci_high: float
    sharpe_point: float
    n_boot: int


def _stationary_bootstrap_indices(
    n: int,
    block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one stationary-bootstrap resample of indices length n.

    Politis & Romano (1994): blocks start at a uniformly random position
    and have geometrically distributed lengths with mean ``block_length``.
    Wrap-around is applied modulo n.

    Parameters
    ----------
    n :
        Length of the original series.
    block_length :
        Expected block length (mean of geometric distribution).
    rng :
        Seeded NumPy random generator.

    Returns
    -------
    np.ndarray of int, shape (n,)
    """
    p = 1.0 / block_length  # probability of ending a block
    indices = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        # Geometric block length: draw until done or n filled
        j = 0
        while i < n:
            indices[i] = (start + j) % n
            i += 1
            j += 1
            if rng.random() < p:
                break  # start a new block
    return indices


def bootstrap_sharpe_ci(
    equity: pd.Series,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    block_length: Optional[float] = None,
) -> BootstrapCI:
    """Stationary bootstrap confidence interval for the Sharpe ratio.

    Sharpe ratios (point estimate and CI bounds) are in daily units
    (mean/std, ddof=1).

    Parameters
    ----------
    equity :
        Equity curve.
    n_boot :
        Number of bootstrap replications.
    seed :
        Seed for reproducibility.
    alpha :
        Significance level; CI = (alpha/2, 1-alpha/2) percentiles.
    block_length :
        Expected block length for stationary bootstrap.
        Default = max(1, sqrt(n)).

    Returns
    -------
    BootstrapCI
    """
    returns = _returns_from_equity(equity)
    n = len(returns)

    sharpe_point = _daily_sharpe(returns)

    if n < 2 or math.isnan(sharpe_point):
        return BootstrapCI(
            ci_low=math.nan,
            ci_high=math.nan,
            sharpe_point=sharpe_point,
            n_boot=n_boot,
        )

    if block_length is None:
        bl = max(1.0, math.sqrt(n))
    else:
        bl = float(block_length)

    rng = np.random.default_rng(seed)
    boot_sharpes = np.empty(n_boot)

    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, bl, rng)
        r_boot = returns[idx]
        boot_sharpes[b] = _daily_sharpe(r_boot)

    # Remove NaN replicates before computing percentiles
    valid = boot_sharpes[~np.isnan(boot_sharpes)]
    if len(valid) == 0:
        return BootstrapCI(
            ci_low=math.nan,
            ci_high=math.nan,
            sharpe_point=sharpe_point,
            n_boot=n_boot,
        )

    ci_low  = float(np.percentile(valid, 100.0 * alpha / 2.0))
    ci_high = float(np.percentile(valid, 100.0 * (1.0 - alpha / 2.0)))

    return BootstrapCI(
        ci_low=ci_low,
        ci_high=ci_high,
        sharpe_point=sharpe_point,
        n_boot=n_boot,
    )
