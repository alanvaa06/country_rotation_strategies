"""Scorecard aggregator for strategy validation (Task B4).

Runs every validation protocol in one shot and returns a :class:`ValidationReport`
whose :attr:`~ValidationReport.verdict` summarises pass/fail against
:class:`Thresholds`.

Design
------
* A single call to ``Engine`` produces the actual equity curve and daily returns.
* The equal-weight buy-and-hold benchmark is built from the same country columns
  (``scores.columns ∩ prices.columns``, excluding ``cfg.bmk``).
* NaN comparisons are always ``False`` (NaN-safe guards via :func:`_safe_ge`,
  :func:`_safe_le`, :func:`_safe_gt`).
* All result types are frozen dataclasses; ``ValidationReport`` holds all
  intermediate results so callers can inspect any sub-metric.

Pure numpy/pandas; no IO; deterministic under fixed ``seed``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from country_rotation.backtest.benchmarks import equal_weight_buy_hold
from country_rotation.backtest.engine import Engine
from country_rotation.config import BacktestConfig
from country_rotation.validation.protocols import (
    MonteCarloNull,
    StabilitySummary,
    SweepResult,
    WalkForwardResult,
    equity_curve,
    monte_carlo_null,
    parameter_sweep,
    stability_summary,
    walk_forward,
)
from country_rotation.validation.statistics import (
    BootstrapCI,
    DSRResult,
    NWResult,
    PSRResult,
    SharpeSignificance,
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    newey_west_tstat,
    probabilistic_sharpe_ratio,
    sharpe_significance,
)

__all__ = [
    "Thresholds",
    "Verdict",
    "ValidationReport",
    "compute_validation",
]


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Thresholds:
    """Evidence thresholds (design policy, spec §7 / validation_formulas.md).

    Attributes
    ----------
    dsr : float
        Deflated Sharpe Ratio must be >= this (default 0.95).
    psr : float
        Probabilistic Sharpe Ratio must be >= this (default 0.95).
    mc_p : float
        Monte-Carlo null p-value must be <= this (default 0.05).
    wfe : float
        Walk-forward efficiency (OOS/IS sharpe ratio) must be >= this
        (default 0.5).
    frac_oos_positive : float
        Fraction of WF folds with positive OOS Sharpe must be >= this
        (default 0.5).
    sharpe_t : float
        Lo (2002) Sharpe t-statistic must be >= this (default 2.0).
    stability_frac_positive : float
        Fraction of sweep configs with positive Sharpe must be >= this
        (default 0.7).
    stability_max_zscore : float
        Absolute Z-score of default config vs sweep distribution must be
        <= this (default 1.5).
    """

    dsr: float = 0.95
    psr: float = 0.95
    mc_p: float = 0.05
    wfe: float = 0.5
    frac_oos_positive: float = 0.5
    sharpe_t: float = 2.0
    stability_frac_positive: float = 0.7
    stability_max_zscore: float = 1.5


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    """Pass/fail summary of the validation scorecard.

    Attributes
    ----------
    checks : dict
        Mapping of check name -> bool.  Keys:
        ``no_overfitting``, ``param_stable``, ``statistically_significant``.
    overall : bool
        True iff all checks pass.
    notes : tuple
        Human-readable one-liner for each failed sub-check.
    """

    checks: dict
    overall: bool
    notes: tuple


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationReport:
    """Full evidence report from :func:`compute_validation`.

    Every field is a frozen result dataclass from the underlying statistics /
    protocol modules so callers can inspect raw numbers.
    """

    sharpe: SharpeSignificance
    psr: PSRResult
    dsr: DSRResult
    nw_vs_eqw: NWResult
    bootstrap: BootstrapCI
    sweep: SweepResult
    stability: StabilitySummary
    walkforward: WalkForwardResult
    mc: MonteCarloNull
    verdict: Verdict


# ---------------------------------------------------------------------------
# NaN-safe comparison helpers
# ---------------------------------------------------------------------------

def _safe_ge(value: float, threshold: float) -> bool:
    """Return ``value >= threshold``; NaN comparisons are always False."""
    if not math.isfinite(value):
        return False
    return value >= threshold


def _safe_le(value: float, threshold: float) -> bool:
    """Return ``value <= threshold``; NaN comparisons are always False."""
    if not math.isfinite(value):
        return False
    return value <= threshold


def _safe_gt(value: float, threshold: float) -> bool:
    """Return ``value > threshold``; NaN comparisons are always False."""
    if not math.isfinite(value):
        return False
    return value > threshold


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_validation(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    grid: dict,
    thresholds: Thresholds = Thresholds(),
    n_boot: int = 500,
    n_mc: int = 100,
    n_folds: int = 5,
    seed: int = 0,
    basis: str = "absolute",
) -> ValidationReport:
    """Run the full validation suite and assemble a :class:`ValidationReport`.

    Parameters
    ----------
    scores :
        Normalized country scores (dates × countries, no benchmark column).
    prices :
        Price levels (dates × countries + benchmark).
    cfg :
        Base :class:`BacktestConfig` for the strategy under test.
    grid :
        Parameter grid for sweep / walk-forward
        (``field_name -> tuple of values``).
    thresholds :
        Evidence thresholds for the verdict checks.
    n_boot :
        Bootstrap replications for the Sharpe CI.
    n_mc :
        Monte-Carlo null simulations.
    n_folds :
        Number of anchored walk-forward folds.
    seed :
        Seed for deterministic bootstrap and Monte-Carlo runs.
    basis :
        Certification basis for the headline statistics.  ``"absolute"``
        (default) computes Sharpe-significance / PSR / DSR / bootstrap / MC on
        the ABSOLUTE net return of the long-only fully-invested book — heavily
        beta-driven.  ``"active"`` computes them on the strategy's ACTIVE daily
        return ``(daily_returns - daily_bmk_returns)`` so they measure
        selection skill (the annualized active Sharpe is the Information
        Ratio).  The parameter sweep, stability summary, DSR trial Sharpes, and
        Monte-Carlo null all switch to the active basis too.

        ``nw_vs_eqw`` (active strategy vs equal-weight null) is unchanged in
        either mode — it is already an active comparison.  ``walk_forward``
        stays on the absolute basis in both modes: it is the OOS-consistency
        gate and selects on IS net Sharpe; converting it is out of scope.

    Returns
    -------
    ValidationReport
    """
    # ------------------------------------------------------------------
    # 1. Run strategy engine; build the headline equity for the basis
    # ------------------------------------------------------------------
    result = Engine(scores, prices, cfg).run()
    net_returns = result.period_results["portfolio_return_net"].dropna()

    # Daily strategy returns (None if engine produced no daily curve)
    strategy_daily: pd.Series = (
        result.daily_returns if result.daily_returns is not None
        else pd.Series(dtype=float)
    )

    if basis == "active":
        # ACTIVE daily = strategy daily minus benchmark daily (common index).
        bmk_daily: pd.Series = (
            result.daily_bmk_returns if result.daily_bmk_returns is not None
            else pd.Series(dtype=float)
        )
        active_idx = strategy_daily.index.intersection(bmk_daily.index)
        active_daily = (
            strategy_daily.loc[active_idx] - bmk_daily.loc[active_idx]
        ).dropna()
        equity = equity_curve(active_daily)
    else:
        equity = equity_curve(net_returns)

    # ------------------------------------------------------------------
    # 2. Equal-weight buy-and-hold null; Newey-West t-stat of excess
    # ------------------------------------------------------------------
    # Country columns = scores.columns ∩ prices.columns (exclude cfg.bmk)
    country_cols = [
        c for c in scores.columns
        if c in prices.columns and c != cfg.bmk
    ]
    eqw_equity = equal_weight_buy_hold(prices[country_cols])
    eqw_daily = eqw_equity.pct_change().dropna()

    # Align on common index for the difference series
    common_idx = strategy_daily.index.intersection(eqw_daily.index)
    if len(common_idx) >= 2:
        excess_daily = strategy_daily.loc[common_idx] - eqw_daily.loc[common_idx]
    else:
        excess_daily = pd.Series(dtype=float)

    nw_vs_eqw = newey_west_tstat(excess_daily)

    # ------------------------------------------------------------------
    # 3. Parameter sweep + stability + DSR (trial sharpes from sweep table)
    # ------------------------------------------------------------------
    sweep = parameter_sweep(scores, prices, cfg, grid, basis=basis)
    stability = stability_summary(sweep, cfg)

    trial_sharpes = sweep.table["sharpe_daily"].to_numpy(dtype=float)
    # Remove NaN trial sharpes before passing to DSR
    trial_sharpes_valid = trial_sharpes[~pd.isna(trial_sharpes)]
    dsr = deflated_sharpe_ratio(equity, trial_sharpes_valid)

    # ------------------------------------------------------------------
    # 4. Walk-forward, Monte-Carlo null, PSR, Sharpe significance, bootstrap
    # ------------------------------------------------------------------
    wf = walk_forward(scores, prices, cfg, grid, n_folds=n_folds)
    mc = monte_carlo_null(scores, prices, cfg, n_sims=n_mc, seed=seed, basis=basis)
    psr = probabilistic_sharpe_ratio(equity, benchmark_sharpe=0.0)
    sharpe = sharpe_significance(equity)
    bootstrap = bootstrap_sharpe_ci(equity, n_boot=n_boot, seed=seed)

    # ------------------------------------------------------------------
    # 5. Verdict checks (NaN-safe; NaN comparison -> False)
    # ------------------------------------------------------------------
    th = thresholds
    notes_list: list[str] = []

    # no_overfitting
    dsr_pass = _safe_ge(dsr.dsr, th.dsr)
    wfe_pass = _safe_ge(wf.wf_efficiency, th.wfe)
    frac_oos_pass = _safe_ge(wf.frac_folds_oos_positive, th.frac_oos_positive)
    mc_pass = _safe_le(mc.p_value, th.mc_p)
    no_overfitting = bool(dsr_pass and wfe_pass and frac_oos_pass and mc_pass)
    if not dsr_pass:
        notes_list.append(
            f"DSR {dsr.dsr:.3f} < threshold {th.dsr}"
        )
    if not wfe_pass:
        notes_list.append(
            f"WF efficiency {wf.wf_efficiency:.3f} < threshold {th.wfe}"
        )
    if not frac_oos_pass:
        notes_list.append(
            f"Frac OOS-positive folds {wf.frac_folds_oos_positive:.2f} < threshold {th.frac_oos_positive}"
        )
    if not mc_pass:
        notes_list.append(
            f"MC p-value {mc.p_value:.3f} > threshold {th.mc_p}"
        )

    # param_stable
    stab_frac_pass = _safe_ge(stability.frac_positive, th.stability_frac_positive)
    stab_zscore_pass = _safe_le(abs(stability.default_zscore), th.stability_max_zscore)
    param_stable = bool(stab_frac_pass and stab_zscore_pass)
    if not stab_frac_pass:
        notes_list.append(
            f"Stability frac_positive {stability.frac_positive:.2f} < threshold {th.stability_frac_positive}"
        )
    if not stab_zscore_pass:
        notes_list.append(
            f"Stability |default_zscore| {abs(stability.default_zscore):.2f} > threshold {th.stability_max_zscore}"
        )

    # statistically_significant
    t_pass = _safe_ge(sharpe.t_stat, th.sharpe_t)
    psr_pass = _safe_ge(psr.psr, th.psr)
    ci_pass = _safe_gt(bootstrap.ci_low, 0.0)
    nw_pass = _safe_gt(nw_vs_eqw.t_stat, 0.0)
    statistically_significant = bool(t_pass and psr_pass and ci_pass and nw_pass)
    if not t_pass:
        notes_list.append(
            f"Sharpe t-stat {sharpe.t_stat:.2f} < threshold {th.sharpe_t}"
        )
    if not psr_pass:
        notes_list.append(
            f"PSR {psr.psr:.3f} < threshold {th.psr}"
        )
    if not ci_pass:
        notes_list.append(
            f"Bootstrap CI low {bootstrap.ci_low:.4f} <= 0"
        )
    if not nw_pass:
        notes_list.append(
            f"NW t-stat vs EQW {nw_vs_eqw.t_stat:.2f} <= 0"
        )

    overall = bool(no_overfitting and param_stable and statistically_significant)

    verdict = Verdict(
        checks={
            "no_overfitting": no_overfitting,
            "param_stable": param_stable,
            "statistically_significant": statistically_significant,
        },
        overall=overall,
        notes=tuple(notes_list),
    )

    return ValidationReport(
        sharpe=sharpe,
        psr=psr,
        dsr=dsr,
        nw_vs_eqw=nw_vs_eqw,
        bootstrap=bootstrap,
        sweep=sweep,
        stability=stability,
        walkforward=wf,
        mc=mc,
        verdict=verdict,
    )
