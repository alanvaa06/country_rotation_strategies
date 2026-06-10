"""
Backtest engine: exact port of the legacy ``Backtest`` period math
(repo-root ``backtest.py``), driven by :class:`BacktestConfig`.

Parity contract (blend mode)
----------------------------
Every step replicates legacy ``Backtest.run_backtest`` bit-for-bit within
a single process:

* Date grid (legacy ``_filter_dates`` ~275-293):
  ``sorted(scores.index[::-periodicity])`` then intersected with the
  prices index and sorted.
* Period tuples (~176-177): ``[(dates[i-1], dates[i]) for i in 1..]`` —
  select at ``d_sel``, earn the return ``d_sel -> d_ret``.
* Selection (~319-380): absolute = score > threshold at ``d_sel``
  (after ``dropna``); relative = top N by score change over
  ``periodicity`` index rows, falling back to top N by score level when
  no previous date exists.
* Selection ``top_n_level`` (additive, NOT legacy): top N by score
  LEVEL at ``d_sel`` — the canonical Asness-Moskowitz-Pedersen (2013)
  rank-of-signal-level selection.  Works with every weighting branch;
  under Cap_Tilt the bottom-N underweights come from the SAME level
  signal (N lowest levels).
* Weighting (~431-560): Equal = 1/N; Risk_Parity = inverse variance over
  ``risk_parity_lookback`` daily observations EXCLUDING ``d_sel``
  (min 10 lookback dates / 5 clean return rows, several equal-weight
  fallbacks — copied verbatim).
* Blending (~562-598): benchmark gets ``bmk_weight``, active weights are
  scaled by ``1 - bmk_weight``; empty selection -> 100% benchmark.
* Turnover (~600-666): first period = ``1 - w_bmk``; afterwards
  ``sum(|dw|)/2`` over countries EXCLUDING the benchmark, ignoring
  changes below 1e-6.
* Returns (~668-724): simple price return ``d_sel -> d_ret`` per asset
  (0.0 on NaN/zero start); portfolio return = dict-ordered dot product;
  ``net = gross - turnover * tc_bps/10000``.
* ``active_return = gross - bmk_return`` — legacy line ~229 uses GROSS,
  not net.  Replicated as-is for parity.
* Daily curve (~833-887): period weights mapped back to ``d_sel``,
  forward-filled, shifted one day, multiplied by daily pct-change.

Column-name mapping vs legacy ``run_backtest()`` result dict::

    period_results column     legacy dict key      legacy Series name
    ----------------------    -----------------    ----------------------
    portfolio_return_gross    returns              Portfolio_Return_Gross
    portfolio_return_net      returns_net          Portfolio_Return_Net
    bmk_return                benchmark_returns    Benchmark_Return
    active_return             active_return        Active_Return
    turnover                  turnover             Turnover
    transaction_cost          transaction_costs    Transaction_Costs

Modes
-----
* ``blend``  — exact legacy behavior; ``historical_weights`` includes the
  benchmark column and is indexed by the period END date (``d_ret``),
  exactly like legacy ``historical_weights``.
* ``active`` — no blending row: ``bmk_weight`` is ignored for
  construction and the portfolio holds only the selected countries.
  ``period_results`` still reports ``bmk_return`` and ``active_return``.

``cfg.execution_lag_days`` is intentionally IGNORED here: parity with the
legacy engine comes first; execution lag lands in Plan B.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from country_rotation.config import BacktestConfig

_RESULT_COLUMNS = [
    "portfolio_return_gross",
    "portfolio_return_net",
    "bmk_return",
    "active_return",
    "turnover",
    "transaction_cost",
]


@dataclass(frozen=True)
class EngineResult:
    """Immutable backtest output.

    Attributes
    ----------
    period_results : pd.DataFrame
        Indexed by period end date (``d_ret``).  Columns:
        ``portfolio_return_gross``, ``portfolio_return_net``,
        ``bmk_return``, ``active_return``, ``turnover``,
        ``transaction_cost``.
    historical_weights : pd.DataFrame
        Indexed by period end date (``d_ret``, legacy convention);
        columns = countries (+ benchmark in blend mode), NaN -> 0.0.
    daily_returns : pd.Series | None
        Daily portfolio return curve (None when no periods exist).
    daily_bmk_returns : pd.Series | None
        Daily benchmark return curve (None when no periods exist).
    """

    period_results: pd.DataFrame
    historical_weights: pd.DataFrame
    daily_returns: Optional[pd.Series]
    daily_bmk_returns: Optional[pd.Series]


class Engine:
    """Period-rebalanced country rotation backtest (legacy-parity port)."""

    def __init__(
        self,
        normalized_score: pd.DataFrame,
        prices: pd.DataFrame,
        cfg: BacktestConfig,
        base_weights: pd.DataFrame | None = None,
    ) -> None:
        self.normalized_score = normalized_score.copy()
        self.prices = prices.copy()
        self.cfg = cfg
        self.selection_criteria = cfg.selection_criteria.lower()
        self.bmk = cfg.bmk
        self.transaction_cost_decimal = cfg.transaction_cost_bps / 10000.0
        # Cap_Tilt only: dates x countries cap-weight base (rows ~sum to 1).
        self.base_weights = (
            base_weights.sort_index().copy() if base_weights is not None else None
        )

        self._validate_inputs()

        # Country universe excludes the benchmark column (legacy ~105).
        self.countries = [
            col for col in self.normalized_score.columns if col != self.bmk
        ]
        # Legacy re-sorts these indexes inside each helper call; cache the
        # identical sorted lists once (same values, same parity).
        self._score_dates_sorted = sorted(self.normalized_score.index)
        self._price_dates_sorted = sorted(self.prices.index)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def _validate_inputs(self) -> None:
        """Replicates legacy ``_validate_inputs`` (~126-151) + mode check."""
        if self.selection_criteria not in ("absolute", "relative", "top_n_level"):
            raise ValueError(
                "selection_criteria must be 'absolute', 'relative' or "
                "'top_n_level'"
            )
        if self.cfg.weighting_method not in ("Equal", "Risk_Parity", "Cap_Tilt"):
            raise ValueError(
                "weighting_method must be 'Equal', 'Risk_Parity' or 'Cap_Tilt'"
            )
        if self.cfg.weighting_method == "Cap_Tilt":
            if self.cfg.mode != "active":
                raise ValueError("Cap_Tilt weighting requires mode='active'")
            if self.base_weights is None:
                raise ValueError("Cap_Tilt weighting requires base_weights")
            if not 0.0 < self.cfg.active_share <= 1.0:
                raise ValueError("active_share must be in (0, 1]")
        if not 0 <= self.cfg.bmk_weight <= 1:
            raise ValueError("bmk_weight must be between 0 and 1")
        if self.bmk not in self.prices.columns:
            raise ValueError(f"Benchmark '{self.bmk}' not found in prices DataFrame")
        if self.cfg.periodicity <= 0:
            raise ValueError("periodicity must be positive")
        if self.cfg.mode not in ("blend", "active"):
            raise ValueError("mode must be 'blend' or 'active'")
        common = self.normalized_score.index.intersection(self.prices.index)
        if len(common) == 0:
            raise ValueError("No common dates found between normalized_score and prices")

    # ------------------------------------------------------------------
    # date grid
    # ------------------------------------------------------------------
    def _filter_dates(self) -> list:
        """Legacy ``_filter_dates`` (~275-293): score-anchored reverse grid."""
        all_score_dates = self.normalized_score.index
        grid_dates = sorted(all_score_dates[::-self.cfg.periodicity])
        valid_dates = sorted(self.prices.index.intersection(grid_dates))
        return valid_dates

    def _get_previous_date(self, current_date, lookback_days: int):
        """Legacy ``_get_previous_date`` (~382-409): index-row lookback."""
        all_dates = self._score_dates_sorted
        if current_date not in all_dates:
            return None
        current_idx = all_dates.index(current_date)
        target_idx = current_idx - lookback_days
        if target_idx < 0:
            return None
        return all_dates[target_idx]

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------
    def _select_countries(self, d_sel) -> list:
        """Legacy ``_select_countries`` (~295-317)."""
        if d_sel not in self.normalized_score.index:
            return []
        if self.selection_criteria == "absolute":
            return self._select_absolute(d_sel)
        if self.selection_criteria == "top_n_level":
            return self._select_top_n_level(d_sel)
        return self._select_relative(d_sel)

    def _select_absolute(self, d_sel) -> list:
        """Legacy ``_select_absolute`` (~319-344): level > threshold."""
        scores = self.normalized_score.loc[d_sel, self.countries]
        scores = scores.dropna()
        return scores[scores > self.cfg.absolute_selection_score].index.tolist()

    def _relative_signal(self, d_sel) -> pd.Series:
        """Signal series behind ``_select_relative``: score change over
        ``periodicity`` index rows, falling back to score level when no
        previous date exists.  Extracted so Cap_Tilt can rank top AND
        bottom names from the same series; ``_select_relative`` output is
        bit-identical to the pre-refactor code."""
        d_prev = self._get_previous_date(d_sel, self.cfg.periodicity)

        if d_prev is None or d_prev not in self.normalized_score.index:
            # No previous date: fall back to score level.
            scores_current = self.normalized_score.loc[d_sel, self.countries]
            return scores_current.dropna()

        scores_current = self.normalized_score.loc[d_sel, self.countries]
        scores_prev = self.normalized_score.loc[d_prev, self.countries]
        return (scores_current - scores_prev).dropna()

    def _select_relative(self, d_sel) -> list:
        """Legacy ``_select_relative`` (~346-380): top N by score change."""
        signal = self._relative_signal(d_sel)
        return signal.nlargest(self.cfg.relative_selection_score).index.tolist()

    def _select_top_n_level(self, d_sel) -> list:
        """Top N countries by score LEVEL at ``d_sel`` (additive, not legacy).

        Canonical AMP-2013 selection: rank the signal level itself and take
        the ``relative_selection_score`` highest names."""
        scores = self.normalized_score.loc[d_sel, self.countries].dropna()
        return scores.nlargest(self.cfg.relative_selection_score).index.tolist()

    # ------------------------------------------------------------------
    # weighting
    # ------------------------------------------------------------------
    def _construct_active_weights(self, selected_countries: list, current_date) -> dict:
        """Legacy ``_construct_active_weights`` (~411-443)."""
        if len(selected_countries) == 0:
            return {}
        if self.cfg.weighting_method == "Equal":
            weight = 1.0 / len(selected_countries)
            return {country: weight for country in selected_countries}
        return self._calculate_risk_parity_weights(selected_countries, current_date)

    def _equal_fallback(self, selected_countries: list) -> dict:
        weight = 1.0 / len(selected_countries)
        return {country: weight for country in selected_countries}

    def _calculate_risk_parity_weights(self, selected_countries: list, current_date) -> dict:
        """Legacy ``_calculate_risk_parity_weights`` (~445-560), verbatim.

        Inverse-variance weights from the last ``risk_parity_lookback``
        daily observations strictly BEFORE ``current_date``; multiple
        equal-weight fallbacks on insufficient/degenerate data.
        """
        all_dates = self._price_dates_sorted
        if current_date not in all_dates:
            return self._equal_fallback(selected_countries)

        current_idx = all_dates.index(current_date)
        start_idx = max(0, current_idx - self.cfg.risk_parity_lookback)
        if start_idx >= current_idx:
            return self._equal_fallback(selected_countries)

        lookback_dates = all_dates[start_idx:current_idx]
        if len(lookback_dates) < 10:  # Minimum 10 periods (legacy ~491)
            return self._equal_fallback(selected_countries)

        returns_data = {}
        for country in selected_countries:
            if country in self.prices.columns:
                prices_series = self.prices.loc[lookback_dates, country]
                returns_series = prices_series.pct_change().dropna()
                if len(returns_series) > 0:
                    returns_data[country] = returns_series

        if len(returns_data) == 0:
            return self._equal_fallback(selected_countries)

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        if len(returns_df) < 5 or returns_df.shape[1] < len(selected_countries):
            return self._equal_fallback(selected_countries)

        volatilities = returns_df.std()
        inverse_variances = 1.0 / (volatilities ** 2)
        inverse_variances = inverse_variances.replace([np.inf, -np.inf], 0)
        inverse_variances = inverse_variances.fillna(0)
        if inverse_variances.sum() == 0:
            return self._equal_fallback(selected_countries)

        risk_parity_weights = inverse_variances / inverse_variances.sum()
        weights_dict = risk_parity_weights.to_dict()
        for country in selected_countries:
            if country not in weights_dict:
                weights_dict[country] = 0.0

        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            weights_dict = {k: v / total_weight for k, v in weights_dict.items()}
        else:
            weights_dict = self._equal_fallback(selected_countries)
        return weights_dict

    def _construct_blended_weights(self, active_weights: dict) -> dict:
        """Legacy ``_construct_blended_weights`` (~562-598)."""
        blended_weights = {}
        if len(active_weights) == 0:
            blended_weights[self.bmk] = 1.0
            return blended_weights

        blended_weights[self.bmk] = self.cfg.bmk_weight
        active_scale = 1.0 - self.cfg.bmk_weight
        for country, weight in active_weights.items():
            blended_weights[country] = weight * active_scale

        total_weight = sum(blended_weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Weights do not sum to 1.0: {total_weight}")
        return blended_weights

    # ------------------------------------------------------------------
    # Cap_Tilt weighting (benchmark-aware construction; active mode only)
    # ------------------------------------------------------------------
    def _cap_tilt_base(self, d_sel) -> Optional[pd.Series]:
        """Cap-weight base for ``d_sel``: last base_weights row at or
        before ``d_sel``, restricted to countries with a valid score AND a
        valid price at ``d_sel``, NaN/negative entries dropped, renormalized
        to sum 1.  Returns None when no usable base exists (caller falls
        back to the previous period's weights)."""
        history = self.base_weights.loc[:d_sel]
        if len(history) == 0:
            return None
        if d_sel not in self.normalized_score.index or d_sel not in self.prices.index:
            return None
        base = history.iloc[-1]
        scores_row = self.normalized_score.loc[d_sel]
        prices_row = self.prices.loc[d_sel]
        cols = [
            c for c in base.index
            if c in self.countries
            and pd.notna(base[c]) and base[c] >= 0.0
            and c in scores_row.index and pd.notna(scores_row[c])
            and c in prices_row.index and pd.notna(prices_row[c])
        ]
        base = base[cols].astype(float)
        total = base.sum()
        if not total > 0.0:
            return None
        return base / total

    def _selection_signal(self, d_sel) -> pd.Series:
        """One signal series for Cap_Tilt top AND bottom ranking.

        'relative' -> the exact ``_select_relative`` signal (score change,
        score-level fallback).  'absolute' and 'top_n_level' -> score level
        at ``d_sel``, so the bottom set is the symmetric mirror: lowest
        score levels."""
        if d_sel not in self.normalized_score.index:
            return pd.Series(dtype=float)
        if self.selection_criteria in ("absolute", "top_n_level"):
            return self.normalized_score.loc[d_sel, self.countries].dropna()
        return self._relative_signal(d_sel)

    def _signal_top(self, signal: pd.Series) -> list:
        """Top names from the signal series — mirrors the existing
        selection rules: 'relative'/'top_n_level' = top N,
        'absolute' = level > threshold."""
        if self.selection_criteria == "absolute":
            return signal[signal > self.cfg.absolute_selection_score].index.tolist()
        return signal.nlargest(self.cfg.relative_selection_score).index.tolist()

    def _construct_cap_tilt_weights(self, d_sel, previous_weights: dict) -> dict:
        """Cap-index base +/- score tilts; fully invested, long-only.

        * tilt_unit = active_share / N (N = number of top names).
        * bottom-N underweight_i = min(tilt_unit, base_i) — long-only clip.
        * each top name gets total_under / N, so the sum stays exactly 1.
        * no usable base row -> hold the previous period's weights.
        """
        base = self._cap_tilt_base(d_sel)
        if base is None:
            return dict(previous_weights)

        signal = self._selection_signal(d_sel)
        # Tilt only investable names: restrict to the base universe.
        signal = signal[signal.index.intersection(base.index)]

        weights = {c: float(w) for c, w in base.items()}
        top = self._signal_top(signal)
        if len(top) == 0:
            return weights  # nothing selected: hold the cap base untouched

        bottom = signal.drop(labels=top).nsmallest(len(top)).index.tolist()
        tilt_unit = self.cfg.active_share / len(top)

        total_under = 0.0
        for c in bottom:
            under = min(tilt_unit, weights[c])
            weights[c] -= under
            total_under += under
        over = total_under / len(top)
        for c in top:
            weights[c] += over
        return weights

    # ------------------------------------------------------------------
    # turnover / returns
    # ------------------------------------------------------------------
    def _calculate_turnover(self, current_weights: dict, previous_weights: dict) -> dict:
        """Legacy ``_calculate_turnover`` (~600-666).

        Extra guard (active mode only, unreachable in blend): an empty
        first-period portfolio deploys nothing -> turnover 0.0.
        """
        if not current_weights and not previous_weights:
            return {"turnover": 0.0, "bought": [], "sold": [], "trades": {}}

        # First period: full deployment of the non-benchmark sleeve.
        if not previous_weights:
            bought = [c for c in current_weights.keys() if c != self.bmk]
            return {
                "turnover": 1.0 - current_weights.get(self.bmk, 0.0),
                "bought": bought,
                "sold": [],
                "trades": {c: current_weights[c] for c in bought},
            }

        all_countries = set(current_weights.keys()).union(set(previous_weights.keys()))
        all_countries.discard(self.bmk)  # Exclude benchmark from turnover

        weight_changes = {}
        bought = []
        sold = []
        total_absolute_change = 0.0
        for country in all_countries:
            prev_weight = previous_weights.get(country, 0.0)
            curr_weight = current_weights.get(country, 0.0)
            change = curr_weight - prev_weight
            if abs(change) > 1e-6:  # Ignore negligible changes
                weight_changes[country] = change
                total_absolute_change += abs(change)
                if change > 0:
                    bought.append(country)
                else:
                    sold.append(country)

        turnover = total_absolute_change / 2.0
        return {
            "turnover": turnover,
            "bought": bought,
            "sold": sold,
            "trades": weight_changes,
        }

    def _calculate_period_returns(self, d_sel, d_ret) -> dict:
        """Legacy ``_calculate_period_returns`` (~668-700)."""
        if d_sel not in self.prices.index or d_ret not in self.prices.index:
            return {}
        price_start = self.prices.loc[d_sel]
        price_end = self.prices.loc[d_ret]
        period_returns = {}
        for asset in self.prices.columns:
            if pd.notna(price_start[asset]) and pd.notna(price_end[asset]) and price_start[asset] != 0:
                period_returns[asset] = (price_end[asset] - price_start[asset]) / price_start[asset]
            else:
                period_returns[asset] = 0.0
        return period_returns

    @staticmethod
    def _calculate_portfolio_return(weights: dict, period_returns: dict) -> float:
        """Legacy ``_calculate_portfolio_return`` (~702-724): ordered dot product."""
        portfolio_return = 0.0
        for asset, weight in weights.items():
            if asset in period_returns:
                portfolio_return += weight * period_returns[asset]
        return portfolio_return

    # ------------------------------------------------------------------
    # daily curve
    # ------------------------------------------------------------------
    def _daily_returns(self, date_tuples: list, historical_weights: pd.DataFrame):
        """Legacy ``get_daily_returns`` (~833-887): ffill weights, shift(1)."""
        daily_asset_returns = self.prices.pct_change()

        sel_dates = [t[0] for t in date_tuples]
        ret_dates = [t[1] for t in date_tuples]

        daily_weights = pd.DataFrame(
            np.nan, index=daily_asset_returns.index, columns=historical_weights.columns
        )
        for d_sel, d_ret in zip(sel_dates, ret_dates):
            if d_ret in historical_weights.index:
                daily_weights.loc[d_sel] = historical_weights.loc[d_ret]

        daily_weights = daily_weights.ffill()
        # Weights decided at Close(d_sel) earn returns from d_sel + 1.
        daily_weights = daily_weights.shift(1)

        start_date = sel_dates[0]
        daily_weights = daily_weights.loc[start_date:]
        daily_asset_returns = daily_asset_returns.loc[start_date:]

        daily_port_return = (daily_weights * daily_asset_returns).sum(axis=1)
        daily_bmk_return = (
            daily_asset_returns[self.bmk]
            if self.bmk in daily_asset_returns.columns
            else pd.Series(0, index=daily_asset_returns.index)
        )
        return daily_port_return, daily_bmk_return

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> EngineResult:
        """Execute the backtest loop (legacy ``run_backtest`` ~153-273)."""
        dates = self._filter_dates()
        date_tuples = [(dates[i - 1], dates[i]) for i in range(1, len(dates))]

        if not date_tuples:
            empty = pd.DataFrame(columns=_RESULT_COLUMNS)
            return EngineResult(
                period_results=empty,
                historical_weights=pd.DataFrame(),
                daily_returns=None,
                daily_bmk_returns=None,
            )

        gross_list: list[float] = []
        net_list: list[float] = []
        bmk_list: list[float] = []
        active_list: list[float] = []
        turnover_list: list[float] = []
        tc_list: list[float] = []
        weights_list: list[dict] = []
        period_dates: list = []

        previous_weights: dict = {}
        for d_sel, d_ret in date_tuples:
            if self.cfg.weighting_method == "Cap_Tilt":
                # Benchmark-aware book: cap base +/- score tilts (active only).
                weights = self._construct_cap_tilt_weights(d_sel, previous_weights)
            else:
                selected_countries = self._select_countries(d_sel)
                active_weights = self._construct_active_weights(selected_countries, d_sel)

                if self.cfg.mode == "blend":
                    weights = self._construct_blended_weights(active_weights)
                else:  # 'active': no benchmark row, bmk_weight ignored.
                    weights = dict(active_weights)

            turnover_result = self._calculate_turnover(weights, previous_weights)
            turnover = turnover_result["turnover"]
            transaction_cost = turnover * self.transaction_cost_decimal

            period_returns = self._calculate_period_returns(d_sel, d_ret)
            portfolio_return_gross = self._calculate_portfolio_return(weights, period_returns)
            portfolio_return_net = portfolio_return_gross - transaction_cost
            benchmark_return = period_returns.get(self.bmk, 0.0)
            # Legacy (~229): active return uses GROSS portfolio return.
            active_return = portfolio_return_gross - benchmark_return

            gross_list.append(portfolio_return_gross)
            net_list.append(portfolio_return_net)
            bmk_list.append(benchmark_return)
            active_list.append(active_return)
            turnover_list.append(turnover)
            tc_list.append(transaction_cost)
            weights_list.append(weights)
            period_dates.append(d_ret)

            previous_weights = weights.copy()

        period_results = pd.DataFrame(
            {
                "portfolio_return_gross": gross_list,
                "portfolio_return_net": net_list,
                "bmk_return": bmk_list,
                "active_return": active_list,
                "turnover": turnover_list,
                "transaction_cost": tc_list,
            },
            index=period_dates,
        )
        # Legacy convention: weights indexed by period END date (d_ret).
        historical_weights = pd.DataFrame(weights_list, index=period_dates).fillna(0.0)

        daily_returns, daily_bmk_returns = self._daily_returns(date_tuples, historical_weights)

        return EngineResult(
            period_results=period_results,
            historical_weights=historical_weights,
            daily_returns=daily_returns,
            daily_bmk_returns=daily_bmk_returns,
        )
