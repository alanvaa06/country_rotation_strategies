"""Research protocols for strategy validation (Task B2).

Engines that drive :class:`country_rotation.backtest.engine.Engine` to
answer three robustness questions:

* ``parameter_sweep`` / ``stability_summary`` — is performance stable in
  a 1-D neighborhood around the chosen configuration?
* ``walk_forward`` — does in-sample (IS) configuration selection survive
  anchored out-of-sample (OOS) evaluation?  STRICT no-look-ahead: the
  config for fold k is chosen using only data strictly before fold k's
  OOS segment.
* ``monte_carlo_null`` — does the actual signal beat a null distribution
  of seeded random-score signals pushed through the *same* engine
  configuration (selection machinery identical, signal randomized)?

Units note
----------
Sharpe ratios are computed via
:func:`country_rotation.validation.statistics.sharpe_significance` on the
equity curve of **period** (rebalance-frequency) net returns, so
``sharpe_daily`` is per-period and ``sharpe_ann`` applies the sqrt(252)
convention of that module.  These figures are used as *comparable scores*
across configurations / folds / simulations, never as standalone
annualized estimates.

Pure numpy/pandas; no IO; deterministic (seeded ``default_rng``).
All result types are frozen dataclasses.
"""
from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd

from country_rotation.backtest.engine import Engine
from country_rotation.config import BacktestConfig
from country_rotation.validation.statistics import sharpe_significance

_NET_COL = "portfolio_return_net"


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def equity_curve(period_returns: pd.Series) -> pd.Series:
    """Compound period returns into an equity curve starting at 1.0.

    Parameters
    ----------
    period_returns :
        Simple returns per period.  NaNs are dropped.

    Returns
    -------
    pd.Series
        ``cumprod(1 + r)`` with a 1.0 prepended at a synthetic start
        index: one business day before the first return for a
        ``DatetimeIndex``, ``first - 1`` for numeric indexes, otherwise
        a fresh positional index.  Empty/all-NaN input -> ``Series([1.0])``.
    """
    r = period_returns.dropna()
    if len(r) == 0:
        return pd.Series([1.0])

    eq = (1.0 + r).cumprod()
    first = r.index[0]
    if isinstance(r.index, pd.DatetimeIndex):
        start = first - pd.tseries.offsets.BDay(1)
    elif pd.api.types.is_numeric_dtype(r.index):
        start = first - 1
    else:  # unsupported index type: fall back to positional labels
        values = np.concatenate([[1.0], eq.to_numpy(dtype=float)])
        return pd.Series(values)
    return pd.concat([pd.Series([1.0], index=[start]), eq])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _config_key(cfg: BacktestConfig) -> tuple:
    """Deduplication key: tuple of all BacktestConfig field values."""
    return tuple(getattr(cfg, f.name) for f in fields(BacktestConfig))


def _net_period_returns(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    base_weights: pd.DataFrame | None = None,
) -> pd.Series:
    """Run the Engine and return net period returns (indexed by d_ret)."""
    result = Engine(scores, prices, cfg, base_weights=base_weights).run()
    return result.period_results[_NET_COL].dropna()


def _active_daily(result) -> pd.Series:
    """ACTIVE daily return series = strategy daily minus benchmark daily.

    ``active_daily = (daily_returns - daily_bmk_returns).dropna()`` aligned on
    the common index.  An empty Series is returned when either daily series is
    missing (degenerate Engine run with no daily curve).
    """
    sd = result.daily_returns
    bd = result.daily_bmk_returns
    if sd is None or bd is None:
        return pd.Series(dtype=float)
    common = sd.index.intersection(bd.index)
    return (sd.loc[common] - bd.loc[common]).dropna()


def _basis_equity(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    basis: str,
    base_weights: pd.DataFrame | None = None,
) -> pd.Series:
    """Run the Engine once and return the equity curve for the chosen basis.

    ``basis='absolute'`` -> ``equity_curve`` of net period returns (existing
    beta-driven path).  ``basis='active'`` -> ``equity_curve`` of the ACTIVE
    daily series (strategy minus benchmark), so every downstream statistic
    becomes an information-ratio-style (alpha) statistic.
    """
    result = Engine(scores, prices, cfg, base_weights=base_weights).run()
    if basis == "active":
        return equity_curve(_active_daily(result))
    return equity_curve(result.period_results[_NET_COL].dropna())


# ---------------------------------------------------------------------------
# Parameter sweep + stability
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepResult:
    """Output of :func:`parameter_sweep`.

    Attributes
    ----------
    table : pd.DataFrame
        One row per unique configuration: all ``BacktestConfig`` fields
        plus ``sharpe_daily``, ``sharpe_ann``, ``total_return`` and
        ``max_drawdown``.  The base configuration is always the first row.
    base_key : tuple
        ``_config_key`` of the base configuration (all config fields).
    """

    table: pd.DataFrame
    base_key: tuple


def parameter_sweep(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    base: BacktestConfig,
    grid: dict[str, tuple],
    basis: str = "absolute",
    base_weights: pd.DataFrame | None = None,
) -> SweepResult:
    """Run 1-D parameter neighborhoods around ``base``.

    For each ``field -> values`` entry in ``grid``, every value is swapped
    into ``base`` one axis at a time (``replace(base, field=value)``) —
    no cartesian product.  The base config is always included; duplicate
    configurations (e.g. a grid value equal to the base value) are
    deduplicated by the full config key.

    A run that raises is skipped with a ``warnings.warn`` — degenerate
    neighborhoods must not abort the sweep.

    Parameters
    ----------
    basis :
        Certification basis for the per-trial Sharpe.  ``"absolute"``
        (default) uses the equity curve of net period returns — the
        long-only, beta-driven book.  ``"active"`` uses the equity curve of
        the ACTIVE daily series ``(daily_returns - daily_bmk_returns)``, so
        ``sharpe_daily`` / ``sharpe_ann`` measure selection skill (an
        information-ratio-style statistic).  Table shape is identical in both
        modes; only the metric values change.
    base_weights :
        Optional cap-weight base (dates x countries) forwarded to every
        Engine run — required when ``base.weighting_method == 'Cap_Tilt'``.

    Returns
    -------
    SweepResult
    """
    candidates: list[BacktestConfig] = [base]
    for field_name, values in grid.items():
        for value in values:
            candidates.append(replace(base, **{field_name: value}))

    seen: set[tuple] = set()
    rows: list[dict] = []
    for cfg in candidates:
        key = _config_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        try:
            result = Engine(scores, prices, cfg, base_weights=base_weights).run()
        except Exception as exc:  # degenerate config: skip, keep sweeping
            warnings.warn(f"parameter_sweep: skipping config {key}: {exc}")
            continue
        if basis == "active":
            returns = _active_daily(result)
        else:
            returns = result.period_results[_NET_COL].dropna()
        eq = equity_curve(returns)
        sig = sharpe_significance(eq)
        row = {f.name: getattr(cfg, f.name) for f in fields(BacktestConfig)}
        row["sharpe_daily"] = sig.sharpe_daily
        row["sharpe_ann"] = sig.sharpe_ann
        row["total_return"] = float((1.0 + returns).prod() - 1.0)
        row["max_drawdown"] = float((eq / eq.cummax() - 1.0).min())
        rows.append(row)

    return SweepResult(table=pd.DataFrame(rows), base_key=_config_key(base))


@dataclass(frozen=True)
class StabilitySummary:
    """Distributional summary of a sweep's ``sharpe_ann`` column.

    ``default_zscore`` = (default_sharpe - sharpe_mean) / sharpe_std
    (0.0 when the std is zero or undefined).
    """

    n_trials: int
    sharpe_mean: float
    sharpe_std: float
    sharpe_min: float
    sharpe_max: float
    frac_positive: float
    default_sharpe: float
    default_zscore: float


def stability_summary(sweep: SweepResult, base: BacktestConfig) -> StabilitySummary:
    """Summarize sweep stability on the ``sharpe_ann`` column.

    The default row is located by matching every config field to ``base``;
    if it is missing (e.g. the base run raised), the table median is used
    as a fallback.  NaN sharpes count as non-positive in ``frac_positive``.
    """
    table = sweep.table
    sharpes = table["sharpe_ann"].astype(float)

    mask = pd.Series(True, index=table.index)
    for f in fields(BacktestConfig):
        mask &= table[f.name] == getattr(base, f.name)
    if mask.any():
        default_sharpe = float(sharpes[mask].iloc[0])
    else:  # base run was skipped: median fallback
        default_sharpe = float(sharpes.median())

    mean = float(sharpes.mean())
    std = float(sharpes.std())
    if not math.isfinite(std) or std == 0.0:
        zscore = 0.0
    else:
        zscore = (default_sharpe - mean) / std

    return StabilitySummary(
        n_trials=len(table),
        sharpe_mean=mean,
        sharpe_std=std,
        sharpe_min=float(sharpes.min()),
        sharpe_max=float(sharpes.max()),
        frac_positive=float((sharpes > 0.0).mean()),
        default_sharpe=default_sharpe,
        default_zscore=zscore,
    )


# ---------------------------------------------------------------------------
# Anchored walk-forward
# ---------------------------------------------------------------------------

_FOLD_COLUMNS = [
    "fold",
    "oos_start",
    "oos_end",
    "params",
    "is_sharpe_daily",
    "oos_sharpe_daily",
    "n_oos_periods",
]

_MIN_IS_PERIODS = 3


@dataclass(frozen=True)
class WalkForwardResult:
    """Output of :func:`walk_forward`.

    Attributes
    ----------
    fold_table : pd.DataFrame
        One row per evaluated fold: ``fold``, ``oos_start``, ``oos_end``,
        ``params`` (dict of the chosen grid-field values),
        ``is_sharpe_daily``, ``oos_sharpe_daily``, ``n_oos_periods``.
    oos_returns : pd.Series
        All OOS net period returns stitched chronologically.
    wf_efficiency : float
        mean(oos_sharpe_daily) / mean(is_sharpe_daily); NaN when the
        denominator is <= 0 or undefined.
    oos_sharpe_ann : float
        Annualized Sharpe of the stitched OOS return series.
    frac_folds_oos_positive : float
        Fraction of folds with oos_sharpe_daily > 0 (NaN counts as
        non-positive).
    """

    fold_table: pd.DataFrame
    oos_returns: pd.Series
    wf_efficiency: float
    oos_sharpe_ann: float
    frac_folds_oos_positive: float


def _candidate_configs(base: BacktestConfig, grid: dict[str, tuple]) -> list[BacktestConfig]:
    """Cartesian product of the grid applied to ``base``, deduplicated."""
    if not grid:
        return [base]
    names = list(grid)
    seen: set[tuple] = set()
    candidates: list[BacktestConfig] = []
    for combo in itertools.product(*(grid[name] for name in names)):
        cfg = replace(base, **dict(zip(names, combo)))
        key = _config_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cfg)
    return candidates


def walk_forward(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    base: BacktestConfig,
    grid: dict[str, tuple],
    n_folds: int = 5,
    base_weights: pd.DataFrame | None = None,
) -> WalkForwardResult:
    """Anchored walk-forward configuration selection.

    Timeline semantics
    ------------------
    * Usable dates = ``scores.index`` ∩ ``prices.index`` (sorted).
    * The first ``max(2 * base.periodicity, 126)`` usable dates are a
      burn-in reserved as training data only (fold 1 needs history).
    * The remaining dates are split into ``n_folds`` contiguous,
      chronological, near-equal OOS segments.
    * Fold k IS data = everything STRICTLY BEFORE segment k's start; the
      best candidate (cartesian grid over ``base``) is chosen by IS
      ``sharpe_daily`` of net period returns.  Candidates erroring or
      producing fewer than 3 IS periods are skipped; ties keep the first
      candidate in grid order (deterministic).
    * Fold k OOS = run the chosen config on data truncated at segment
      k's END, keeping the period returns whose ``d_ret`` falls inside
      the segment.  A period straddling the boundary (selected before
      the segment, realized inside it) is OOS: its selection used only
      pre-segment information.

    NO LOOK-AHEAD: nothing at or after segment k's start influences
    fold k's config choice; nothing after segment k's end influences
    fold k at all.  ``base_weights`` (Cap_Tilt) is truncated with the SAME
    ``.loc`` masks as prices for both IS and OOS runs, so cap-weight rows
    never leak future data into a fold.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    candidates = _candidate_configs(base, grid)
    usable = scores.index.intersection(prices.index).sort_values()
    burn_in = max(2 * base.periodicity, 126)
    pool = usable[burn_in:]
    if len(pool) < n_folds:
        raise ValueError(
            f"Not enough usable dates ({len(usable)}) after burn-in "
            f"({burn_in}) to form {n_folds} folds"
        )
    segments = np.array_split(np.arange(len(pool)), n_folds)

    rows: list[dict] = []
    oos_parts: list[pd.Series] = []
    for k, seg_positions in enumerate(segments, start=1):
        seg_dates = pool[seg_positions]
        seg_start, seg_end = seg_dates[0], seg_dates[-1]

        # --- IS: choose config on data strictly before the segment -----
        is_scores = scores.loc[scores.index < seg_start]
        is_prices = prices.loc[prices.index < seg_start]
        is_base_weights = (
            base_weights.loc[base_weights.index < seg_start]
            if base_weights is not None
            else None
        )
        best_cfg = None
        best_sharpe = -math.inf
        for cand in candidates:
            try:
                net = _net_period_returns(
                    is_scores, is_prices, cand, base_weights=is_base_weights
                )
            except Exception as exc:
                warnings.warn(f"walk_forward: fold {k} IS run failed for "
                              f"{_config_key(cand)}: {exc}")
                continue
            if len(net) < _MIN_IS_PERIODS:
                continue
            sharpe = sharpe_significance(equity_curve(net)).sharpe_daily
            if math.isnan(sharpe):
                continue
            if sharpe > best_sharpe:
                best_cfg, best_sharpe = cand, sharpe
        if best_cfg is None:
            warnings.warn(f"walk_forward: fold {k} has no viable candidate; skipped")
            continue

        # --- OOS: run chosen config truncated at segment end ----------
        try:
            oos_all = _net_period_returns(
                scores.loc[scores.index <= seg_end],
                prices.loc[prices.index <= seg_end],
                best_cfg,
                base_weights=(
                    base_weights.loc[base_weights.index <= seg_end]
                    if base_weights is not None
                    else None
                ),
            )
        except Exception as exc:
            warnings.warn(f"walk_forward: fold {k} OOS run failed: {exc}")
            continue
        oos_net = oos_all[(oos_all.index >= seg_start) & (oos_all.index <= seg_end)]
        oos_sharpe = (
            sharpe_significance(equity_curve(oos_net)).sharpe_daily
            if len(oos_net) > 0
            else math.nan
        )

        rows.append(
            {
                "fold": k,
                "oos_start": seg_start,
                "oos_end": seg_end,
                "params": {name: getattr(best_cfg, name) for name in grid},
                "is_sharpe_daily": best_sharpe,
                "oos_sharpe_daily": oos_sharpe,
                "n_oos_periods": int(len(oos_net)),
            }
        )
        if len(oos_net) > 0:
            oos_parts.append(oos_net)

    fold_table = pd.DataFrame(rows, columns=_FOLD_COLUMNS)
    oos_returns = (
        pd.concat(oos_parts).sort_index() if oos_parts else pd.Series(dtype=float)
    )

    if len(fold_table) == 0:
        return WalkForwardResult(
            fold_table=fold_table,
            oos_returns=oos_returns,
            wf_efficiency=math.nan,
            oos_sharpe_ann=math.nan,
            frac_folds_oos_positive=math.nan,
        )

    is_mean = float(fold_table["is_sharpe_daily"].astype(float).mean())
    oos_mean = float(fold_table["oos_sharpe_daily"].astype(float).mean())
    if not math.isfinite(is_mean) or is_mean <= 0.0:
        wf_efficiency = math.nan
    else:
        wf_efficiency = oos_mean / is_mean

    oos_sharpe_ann = (
        sharpe_significance(equity_curve(oos_returns)).sharpe_ann
        if len(oos_returns) > 0
        else math.nan
    )

    return WalkForwardResult(
        fold_table=fold_table,
        oos_returns=oos_returns,
        wf_efficiency=wf_efficiency,
        oos_sharpe_ann=oos_sharpe_ann,
        frac_folds_oos_positive=float(
            (fold_table["oos_sharpe_daily"].astype(float) > 0.0).mean()
        ),
    )


# ---------------------------------------------------------------------------
# Monte-Carlo random-signal null
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonteCarloNull:
    """Output of :func:`monte_carlo_null`.

    ``p_value`` uses add-one smoothing:
    ``p = (count(null >= actual) + 1) / (n_sims + 1)`` — the actual run
    counts as one member of its own null family, so p is never exactly 0
    (Davison & Hinkley convention).  NaN null sharpes are treated as 0.0;
    a NaN actual sharpe is treated as 0.0 for the comparison (degenerate
    strategies cannot claim significance through NaN propagation).
    """

    actual_sharpe_ann: float
    null_sharpes: np.ndarray
    null_mean: float
    null_q95: float
    p_value: float
    n_sims: int


def monte_carlo_null(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    n_sims: int = 200,
    seed: int = 0,
    basis: str = "absolute",
    base_weights: pd.DataFrame | None = None,
) -> MonteCarloNull:
    """Random-signal null distribution through the same Engine config.

    Simulation ``s`` draws ``default_rng(seed + s).random(scores.shape)``
    as a score DataFrame (same index/columns as ``scores``) and runs the
    identical :class:`BacktestConfig` — only the signal is randomized,
    the selection/weighting/cost machinery is untouched.

    Parameters
    ----------
    basis :
        Certification basis for both the actual and the null Sharpes.
        ``"absolute"`` (default) annualizes the net-period-return equity;
        ``"active"`` annualizes the ACTIVE daily series
        ``(daily_returns - daily_bmk_returns)``, giving an information-ratio
        null.  The random books run through the same active-mode Engine cfg,
        so they too expose ``daily_bmk_returns``.  ``p_value`` (add-one
        smoothing) and the NaN→0 guard are unchanged.
    base_weights :
        Optional cap-weight base (dates x countries) forwarded to the actual
        AND every null Engine run — required for ``Cap_Tilt`` configs, so the
        null holds construction constant and randomizes only the signal.

    Returns
    -------
    MonteCarloNull
    """
    actual = sharpe_significance(
        _basis_equity(scores, prices, cfg, basis, base_weights=base_weights)
    ).sharpe_ann

    null_sharpes = np.empty(n_sims, dtype=float)
    for s in range(n_sims):
        rng = np.random.default_rng(seed + s)
        random_scores = pd.DataFrame(
            rng.random(scores.shape), index=scores.index, columns=scores.columns
        )
        sharpe = sharpe_significance(
            _basis_equity(random_scores, prices, cfg, basis, base_weights=base_weights)
        ).sharpe_ann
        null_sharpes[s] = 0.0 if math.isnan(sharpe) else sharpe

    actual_cmp = 0.0 if math.isnan(actual) else actual
    p_value = (float(np.sum(null_sharpes >= actual_cmp)) + 1.0) / (n_sims + 1.0)

    return MonteCarloNull(
        actual_sharpe_ann=actual,
        null_sharpes=null_sharpes,
        null_mean=float(null_sharpes.mean()),
        null_q95=float(np.quantile(null_sharpes, 0.95)),
        p_value=p_value,
        n_sims=n_sims,
    )
