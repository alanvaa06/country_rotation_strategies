"""
Backtest Module for Country Rotation Strategy

This module contains the Backtest class for running backtests on country rotation strategies.

Author: Alan Vazquez, CFA, Master Jedi
Last Updated: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import spearmanr, pearsonr
from scipy import stats


class Backtest:
    """
    Backtest class for country rotation strategy.
    
    This class implements a robust backtesting framework that:
    - Selects countries based on absolute or relative scoring criteria
    - Constructs portfolio weights using Equal or Risk Parity weighting
    - Blends active portfolio with a benchmark
    - Calculates portfolio returns and active returns (gross and net)
    - Tracks transaction costs and turnover
    - Avoids look-ahead bias by using proper date filtering
    
    Parameters
    ----------
    normalized_score : pd.DataFrame
        DataFrame of normalized scores (0-1) with dates as index and countries as columns
    prices : pd.DataFrame
        DataFrame of historical price levels (not returns) with dates as index and countries as columns
        Note: This DataFrame contains the Benchmark column
    selection_criteria : str, optional
        "absolute" (default) or "relative"
    absolute_selection_score : float, optional
        Threshold for absolute selection (default 0.75)
    relative_selection_score : int, optional
        Number of top countries to select based on score change (default 5)
    weighting_method : str, optional
        Weighting method for active portfolio: "Equal" (default) or "Risk_Parity"
        - "Equal": 1/N weighting for selected countries
        - "Risk_Parity": Inverse volatility weighting for equal risk contribution
    bmk : str, optional
        Column name of the Benchmark within the prices DataFrame (default "Benchmark")
    bmk_weight : float, optional
        The fixed weight allocation for the benchmark (default 0.0)
    periodicity : int, optional
        Rebalancing frequency in days (default 5)
    transaction_cost_bps : float, optional
        Transaction cost in basis points (default 2.0 bps)
    risk_parity_lookback : int, optional
        Lookback window in days for Risk Parity covariance estimation (default 60)
        Only used when weighting_method="Risk_Parity"
    output_folder : str, optional
        Folder path to save output plots (default "outputs/plots")
    save_plots : bool, optional
        Whether to save plots to files (default False). If True, plots are saved to output_folder.
    """
    
    def __init__(
        self,
        normalized_score: pd.DataFrame,
        prices: pd.DataFrame,
        selection_criteria: str = "absolute",
        absolute_selection_score: float = 0.75,
        relative_selection_score: int = 5,
        weighting_method: str = "Equal",
        bmk: str = "Benchmark",
        bmk_weight: float = 0.0,
        periodicity: int = 5,
        transaction_cost_bps: float = 2.0,
        risk_parity_lookback: int = 60,
        output_folder: str = "outputs/plots",
        save_plots: bool = False
    ):
        """Initialize the Backtest class with strategy parameters."""
        # Store input parameters
        self.normalized_score = normalized_score.copy()
        self.prices = prices.copy()
        self.selection_criteria = selection_criteria.lower()
        self.absolute_selection_score = absolute_selection_score
        self.relative_selection_score = relative_selection_score
        self.weighting_method = weighting_method
        self.bmk = bmk
        self.bmk_weight = bmk_weight
        self.periodicity = periodicity
        self.transaction_cost_bps = transaction_cost_bps / 10000.0  # Convert to decimal
        self.risk_parity_lookback = risk_parity_lookback  # Lookback for covariance estimation
        self.output_folder = output_folder
        self.save_plots = save_plots
        
        # Create output folder if saving plots
        if self.save_plots:
            import os
            os.makedirs(self.output_folder, exist_ok=True)
        
        # Validate inputs
        self._validate_inputs()
        
        # Extract country columns (exclude benchmark)
        self.countries = [col for col in self.normalized_score.columns if col != self.bmk]
        
        # Initialize result containers
        self.returns = None  # Gross returns
        self.returns_net = None  # Net returns (after transaction costs)
        self.historical_countries = None
        self.historical_active_weights = None
        self.historical_weights = None
        self.active_return = None
        self.dates = None
        self.date_tuples = None
        self.daily_returns = None
        self.daily_benchmark_returns = None
        
        # Initialize transaction tracking containers
        self.transaction_costs = None
        self.turnover = None
        self.countries_bought = None
        self.countries_sold = None
        self.historical_trades = None
        
    def _validate_inputs(self):
        """Validate input parameters and data."""
        # Check selection criteria
        if self.selection_criteria not in ["absolute", "relative"]:
            raise ValueError("selection_criteria must be 'absolute' or 'relative'")
        
        # Check weighting method
        if self.weighting_method not in ["Equal", "Risk_Parity"]:
            raise ValueError("weighting_method must be 'Equal' or 'Risk_Parity'")
        
        # Check benchmark weight
        if not 0 <= self.bmk_weight <= 1:
            raise ValueError("bmk_weight must be between 0 and 1")
        
        # Check if benchmark exists in prices
        if self.bmk not in self.prices.columns:
            raise ValueError(f"Benchmark '{self.bmk}' not found in prices DataFrame")
        
        # Check periodicity
        if self.periodicity <= 0:
            raise ValueError("periodicity must be positive")
        
        # Check date alignment (dates should be in both dataframes)
        common_dates = self.normalized_score.index.intersection(self.prices.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates found between normalized_score and prices")
    
    def run_backtest(self):
        """
        Execute the backtest loop.
        
        This method:
        1. Filters dates based on periodicity
        2. For each rebalancing period:
           - Selects countries based on criteria
           - Constructs active and blended weights
           - Calculates turnover and transaction costs
           - Calculates portfolio returns (gross and net)
           - Calculates active returns
        3. Stores all results as class attributes
        
        Returns
        -------
        dict
            Dictionary containing all backtest results
        """
        # Step 1: Filter dates by periodicity
        self.dates = self._filter_dates()
        
        # Step 2: Generate date tuples (selection_date, return_date)
        self.date_tuples = [(self.dates[i-1], self.dates[i]) 
                            for i in range(1, len(self.dates))]
        
        # Initialize storage lists
        portfolio_returns_list = []
        portfolio_returns_net_list = []
        benchmark_returns_list = []
        active_returns_list = []
        selected_countries_list = []
        active_weights_list = []
        blended_weights_list = []
        period_dates = []
        
        # Transaction tracking lists
        turnover_list = []
        transaction_costs_list = []
        countries_bought_list = []
        countries_sold_list = []
        trade_details_list = []
        
        # Track previous period weights for turnover calculation
        previous_weights = {}
        
        # Step 3: Loop through each period
        for idx, (d_sel, d_ret) in enumerate(self.date_tuples):
            # Select countries
            selected_countries = self._select_countries(d_sel)
            
            # Construct weights
            active_weights = self._construct_active_weights(selected_countries, current_date=d_sel)
            blended_weights = self._construct_blended_weights(active_weights)
            
            # Calculate turnover and transaction costs
            turnover_result = self._calculate_turnover(blended_weights, previous_weights)
            turnover = turnover_result['turnover']
            transaction_cost = turnover * self.transaction_cost_bps
            bought = turnover_result['bought']
            sold = turnover_result['sold']
            trades = turnover_result['trades']
            
            # Calculate returns for the period
            period_returns = self._calculate_period_returns(d_sel, d_ret)
            
            # Calculate gross portfolio return
            portfolio_return_gross = self._calculate_portfolio_return(blended_weights, period_returns)
            
            # Calculate net portfolio return (after transaction costs)
            portfolio_return_net = portfolio_return_gross - transaction_cost
            
            # Calculate benchmark return
            benchmark_return = period_returns.get(self.bmk, 0.0)
            
            # Calculate active return (using gross returns)
            active_return = portfolio_return_gross - benchmark_return
            
            # Store results
            portfolio_returns_list.append(portfolio_return_gross)
            portfolio_returns_net_list.append(portfolio_return_net)
            benchmark_returns_list.append(benchmark_return)
            active_returns_list.append(active_return)
            selected_countries_list.append(selected_countries)
            active_weights_list.append(active_weights)
            blended_weights_list.append(blended_weights)
            period_dates.append(d_ret)
            
            # Store transaction tracking data
            turnover_list.append(turnover)
            transaction_costs_list.append(transaction_cost)
            countries_bought_list.append(bought)
            countries_sold_list.append(sold)
            trade_details_list.append(trades)
            
            # Update previous weights for next iteration
            previous_weights = blended_weights.copy()
        
        # Step 4: Convert results to DataFrames and Series
        self._store_results(
            period_dates,
            portfolio_returns_list,
            portfolio_returns_net_list,
            benchmark_returns_list,
            active_returns_list,
            selected_countries_list,
            active_weights_list,
            blended_weights_list,
            turnover_list,
            transaction_costs_list,
            countries_bought_list,
            countries_sold_list,
            trade_details_list
        )
        
        # Step 5: Calculate Daily Returns
        daily_port, daily_bmk = self.get_daily_returns()
        self.daily_returns = daily_port
        self.daily_benchmark_returns = daily_bmk

        return self.get_results()
    
    def _filter_dates(self):
        """
        Filter dates using periodicity, anchored to normalized scores history.
        
        Returns
        -------
        list
            Sorted list of dates at specified periodicity intervals
        """
        # 1. Define master grid from Normalized Scores (anchored at end)
        # User preference: Use normalized_score index for date determination
        all_score_dates = self.normalized_score.index
        grid_dates = sorted(all_score_dates[::-self.periodicity])
        
        # 2. Filter for price availability
        # We can only trade if we have prices (to calculate returns/execute trades)
        valid_dates = sorted(self.prices.index.intersection(grid_dates))
        
        return valid_dates
    
    def _select_countries(self, d_sel):
        """
        Select countries based on the selection criteria.
        
        Parameters
        ----------
        d_sel : date
            Selection/rebalancing date
        
        Returns
        -------
        list
            List of selected country names
        """
        if d_sel not in self.normalized_score.index:
            return []
        
        if self.selection_criteria == "absolute":
            return self._select_absolute(d_sel)
        elif self.selection_criteria == "relative":
            return self._select_relative(d_sel)
        else:
            raise ValueError(f"Unknown selection criteria: {self.selection_criteria}")
    
    def _select_absolute(self, d_sel):
        """
        Select countries using absolute threshold method.
        
        Parameters
        ----------
        d_sel : date
            Selection date
        
        Returns
        -------
        list
            List of countries where score > absolute_selection_score
        """
        # Get scores at selection date
        scores = self.normalized_score.loc[d_sel, self.countries]
        
        # Handle missing data
        scores = scores.dropna()
        
        # Select countries above threshold
        selected = scores[scores > self.absolute_selection_score].index.tolist()
        
        # Edge case: If no countries meet threshold, return empty list
        # (will default to 100% benchmark in weight construction)
        return selected
    
    def _select_relative(self, d_sel):
        """
        Select countries using relative ranking method based on score change.
        
        Parameters
        ----------
        d_sel : date
            Selection date
        
        Returns
        -------
        list
            List of top N countries with highest positive score change
        """
        # Find previous date (period days ago)
        d_prev = self._get_previous_date(d_sel, self.periodicity)
        
        if d_prev is None or d_prev not in self.normalized_score.index:
            # If no previous date available, use current scores
            scores_current = self.normalized_score.loc[d_sel, self.countries]
            scores_current = scores_current.dropna()
            top_countries = scores_current.nlargest(self.relative_selection_score).index.tolist()
            return top_countries
        
        # Calculate score change
        scores_current = self.normalized_score.loc[d_sel, self.countries]
        scores_prev = self.normalized_score.loc[d_prev, self.countries]
        
        # Handle missing data
        score_change = (scores_current - scores_prev).dropna()
        
        # Select top N countries with highest score change
        top_countries = score_change.nlargest(self.relative_selection_score).index.tolist()
        
        return top_countries
    
    def _get_previous_date(self, current_date, lookback_days):
        """
        Get the date that is lookback_days before current_date.
        
        Parameters
        ----------
        current_date : date
            Current date
        lookback_days : int
            Number of days to look back
        
        Returns
        -------
        date or None
            Previous date or None if not found
        """
        all_dates = sorted(self.normalized_score.index)
        
        if current_date not in all_dates:
            return None
        
        current_idx = all_dates.index(current_date)
        target_idx = current_idx - lookback_days
        
        if target_idx < 0:
            return None
        
        return all_dates[target_idx]
    
    def _construct_active_weights(self, selected_countries, current_date=None):
        """
        Construct active portfolio weights using the specified weighting method.
        
        Parameters
        ----------
        selected_countries : list
            List of selected country names
        current_date : datetime, optional
            Current date for Risk Parity covariance calculation
        
        Returns
        -------
        dict
            Dictionary mapping country names to weights
        """
        if len(selected_countries) == 0:
            # No countries selected - return empty weights
            return {}
        
        if self.weighting_method == "Equal":
            # Equal weighting: 1/N for each country
            weight = 1.0 / len(selected_countries)
            return {country: weight for country in selected_countries}
        
        elif self.weighting_method == "Risk_Parity":
            # Risk Parity weighting: equal risk contribution
            if current_date is None:
                raise ValueError("Risk Parity requires current_date parameter")
            return self._calculate_risk_parity_weights(selected_countries, current_date)
        
        else:
            raise ValueError(f"Unsupported weighting method: {self.weighting_method}")
    
    def _calculate_risk_parity_weights(self, selected_countries, current_date):
        """
        Calculate Risk Parity weights based on inverse variance.
        
        Risk Parity allocates weights so that each asset contributes equally to portfolio risk.
        Uses inverse variance weighting (w_i ∝ 1/σ_i²) as an approximation to equal risk contribution.
        Uses a lookback window to estimate volatility from historical returns (excluding current date).
        
        Note: This uses the simplified approach assuming uncorrelated assets. For highly correlated
        assets, true risk parity would require iterative optimization, but inverse variance provides
        a good approximation while being computationally efficient.
        
        Parameters
        ----------
        selected_countries : list
            List of selected country names
        current_date : datetime
            Current date for calculating historical returns (not included in lookback)
        
        Returns
        -------
        dict
            Dictionary mapping country names to risk parity weights
        """
        # Get historical prices for lookback period
        all_dates = sorted(self.prices.index)
        
        # Find current date index
        if current_date not in all_dates:
            # Fallback to equal weights if date not found
            weight = 1.0 / len(selected_countries)
            return {country: weight for country in selected_countries}
        
        current_idx = all_dates.index(current_date)
        
        # Calculate start index for lookback
        start_idx = max(0, current_idx - self.risk_parity_lookback)
        
        if start_idx >= current_idx:
            # Not enough historical data, fallback to equal weights
            weight = 1.0 / len(selected_countries)
            return {country: weight for country in selected_countries}
        
        # Get historical prices for selected countries
        lookback_dates = all_dates[start_idx:current_idx]
        
        if len(lookback_dates) < 10:  # Minimum 10 periods for meaningful estimate
            # Fallback to equal weights
            weight = 1.0 / len(selected_countries)
            return {country: weight for country in selected_countries}
        
        # Calculate returns for selected countries
        returns_data = {}
        for country in selected_countries:
            if country in self.prices.columns:
                prices_series = self.prices.loc[lookback_dates, country]
                # Calculate returns
                returns_series = prices_series.pct_change().dropna()
                if len(returns_series) > 0:
                    returns_data[country] = returns_series
        
        if len(returns_data) == 0:
            # No valid return data, fallback to equal weights
            weight = 1.0 / len(selected_countries)
            return {country: weight for country in selected_countries}
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 5 or returns_df.shape[1] < len(selected_countries):
            # Insufficient data, fallback to equal weights
            weight = 1.0 / len(selected_countries)
            return {country: weight for country in selected_countries}
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov()
        
        # Calculate volatilities (standard deviations)
        volatilities = returns_df.std()
        
        # True Risk Parity: Inverse variance weighting
        # For equal risk contribution, weight is proportional to 1/variance (1/σ²)
        # This ensures each asset contributes equally to total portfolio risk
        inverse_variances = 1.0 / (volatilities ** 2)
        
        # Handle edge cases (zero or near-zero volatility)
        inverse_variances = inverse_variances.replace([np.inf, -np.inf], 0)
        inverse_variances = inverse_variances.fillna(0)
        
        if inverse_variances.sum() == 0:
            # All volatilities are zero or invalid, fallback to equal weights
            weight = 1.0 / len(selected_countries)
            return {country: weight for country in selected_countries}
        
        # Normalize to sum to 1
        risk_parity_weights = inverse_variances / inverse_variances.sum()
        
        # Convert to dictionary
        weights_dict = risk_parity_weights.to_dict()
        
        # Ensure all selected countries are in the dictionary (even if missing data)
        for country in selected_countries:
            if country not in weights_dict:
                weights_dict[country] = 0.0
        
        # Renormalize to ensure sum = 1.0
        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            weights_dict = {k: v / total_weight for k, v in weights_dict.items()}
        else:
            # Fallback to equal weights
            weight = 1.0 / len(selected_countries)
            weights_dict = {country: weight for country in selected_countries}
        
        return weights_dict
    
    def _construct_blended_weights(self, active_weights):
        """
        Construct final blended weights including benchmark.
        
        Formula: Weight_Final = (bmk_weight * Benchmark) + ((1 - bmk_weight) * Active_Portfolio)
        
        Parameters
        ----------
        active_weights : dict
            Dictionary of active portfolio weights
        
        Returns
        -------
        dict
            Dictionary of final blended weights (including benchmark)
        """
        blended_weights = {}
        
        # If no active countries selected, allocate 100% to benchmark
        if len(active_weights) == 0:
            blended_weights[self.bmk] = 1.0
            return blended_weights
        
        # Add benchmark weight
        blended_weights[self.bmk] = self.bmk_weight
        
        # Scale active weights by (1 - bmk_weight) and add to blended weights
        active_scale = 1.0 - self.bmk_weight
        for country, weight in active_weights.items():
            blended_weights[country] = weight * active_scale
        
        # Verify weights sum to 1.0 (with small tolerance for floating point errors)
        total_weight = sum(blended_weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Weights do not sum to 1.0: {total_weight}")
        
        return blended_weights
    
    def _calculate_turnover(self, current_weights, previous_weights):
        """
        Calculate portfolio turnover and track traded countries.
        
        Turnover = sum of absolute weight changes / 2
        (divide by 2 because buying = selling in terms of dollar volume)
        
        Parameters
        ----------
        current_weights : dict
            Current period portfolio weights
        previous_weights : dict
            Previous period portfolio weights
        
        Returns
        -------
        dict
            Dictionary containing:
            - turnover: float (total turnover)
            - bought: list (countries with increased positions)
            - sold: list (countries with decreased positions)
            - trades: dict (weight changes by country)
        """
        # If first period, consider it as full deployment (100% turnover)
        if not previous_weights:
            bought = [c for c in current_weights.keys() if c != self.bmk]
            return {
                'turnover': 1.0 - current_weights.get(self.bmk, 0.0),  # Exclude benchmark from turnover
                'bought': bought,
                'sold': [],
                'trades': {c: current_weights[c] for c in bought}
            }
        
        # Get all countries that appear in either period (excluding benchmark)
        all_countries = set(current_weights.keys()).union(set(previous_weights.keys()))
        all_countries.discard(self.bmk)  # Exclude benchmark from turnover calculation
        
        # Calculate weight changes
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
        
        # Turnover is half of total absolute change
        # (because every dollar sold is matched by a dollar bought)
        turnover = total_absolute_change / 2.0
        
        return {
            'turnover': turnover,
            'bought': bought,
            'sold': sold,
            'trades': weight_changes
        }
    
    def _calculate_period_returns(self, d_sel, d_ret):
        """
        Calculate simple returns for all assets between d_sel and d_ret.
        
        Parameters
        ----------
        d_sel : date
            Selection date (start of period)
        d_ret : date
            Return date (end of period)
        
        Returns
        -------
        dict
            Dictionary mapping asset names to their period returns
        """
        # Check if dates exist in prices
        if d_sel not in self.prices.index or d_ret not in self.prices.index:
            return {}
        
        # Get prices at both dates
        price_start = self.prices.loc[d_sel]
        price_end = self.prices.loc[d_ret]
        
        # Calculate simple returns: (P_end - P_start) / P_start
        period_returns = {}
        for asset in self.prices.columns:
            if pd.notna(price_start[asset]) and pd.notna(price_end[asset]) and price_start[asset] != 0:
                period_returns[asset] = (price_end[asset] - price_start[asset]) / price_start[asset]
            else:
                period_returns[asset] = 0.0
        
        return period_returns
    
    def _calculate_portfolio_return(self, weights, period_returns):
        """
        Calculate portfolio return as dot product of weights and returns.
        
        Parameters
        ----------
        weights : dict
            Dictionary of portfolio weights
        period_returns : dict
            Dictionary of period returns
        
        Returns
        -------
        float
            Portfolio return for the period
        """
        portfolio_return = 0.0
        
        for asset, weight in weights.items():
            if asset in period_returns:
                portfolio_return += weight * period_returns[asset]
        
        return portfolio_return
    
    def _store_results(
        self,
        dates,
        portfolio_returns,
        portfolio_returns_net,
        benchmark_returns,
        active_returns,
        selected_countries,
        active_weights,
        blended_weights,
        turnover_list,
        transaction_costs_list,
        countries_bought_list,
        countries_sold_list,
        trade_details_list
    ):
        """
        Convert results to pandas objects and store as class attributes.
        
        Parameters
        ----------
        dates : list
            List of period end dates
        portfolio_returns : list
            List of gross portfolio returns
        portfolio_returns_net : list
            List of net portfolio returns (after transaction costs)
        benchmark_returns : list
            List of benchmark returns
        active_returns : list
            List of active returns
        selected_countries : list of lists
            List of selected countries for each period
        active_weights : list of dicts
            List of active weight dictionaries
        blended_weights : list of dicts
            List of blended weight dictionaries
        turnover_list : list
            List of turnover values
        transaction_costs_list : list
            List of transaction costs
        countries_bought_list : list of lists
            List of countries bought each period
        countries_sold_list : list of lists
            List of countries sold each period
        trade_details_list : list of dicts
            List of detailed trade information
        """
        # Store returns as Series (gross returns for performance analysis)
        self.returns = pd.Series(portfolio_returns, index=dates, name='Portfolio_Return_Gross')
        self.returns_net = pd.Series(portfolio_returns_net, index=dates, name='Portfolio_Return_Net')
        self.benchmark_returns = pd.Series(benchmark_returns, index=dates, name='Benchmark_Return')
        self.active_return = pd.Series(active_returns, index=dates, name='Active_Return')
        
        # Store transaction tracking data
        self.turnover = pd.Series(turnover_list, index=dates, name='Turnover')
        self.transaction_costs = pd.Series(transaction_costs_list, index=dates, name='Transaction_Costs')
        
        # Store historical countries as DataFrame
        # Create a DataFrame where each row is a date and columns are country positions (0, 1, 2, ...)
        max_countries = max(len(countries) for countries in selected_countries) if selected_countries else 0
        countries_data = []
        for countries in selected_countries:
            row = countries + [None] * (max_countries - len(countries))
            countries_data.append(row)
        
        self.historical_countries = pd.DataFrame(
            countries_data,
            index=dates,
            columns=range(max_countries)
        )
        
        # Store historical active weights as DataFrame
        self.historical_active_weights = pd.DataFrame(active_weights, index=dates).fillna(0.0)
        
        # Store historical blended weights as DataFrame
        self.historical_weights = pd.DataFrame(blended_weights, index=dates).fillna(0.0)
        
        # Store countries bought and sold
        max_bought = max(len(bought) for bought in countries_bought_list) if countries_bought_list else 0
        max_sold = max(len(sold) for sold in countries_sold_list) if countries_sold_list else 0
        
        bought_data = []
        for bought in countries_bought_list:
            row = bought + [None] * (max_bought - len(bought))
            bought_data.append(row)
        
        sold_data = []
        for sold in countries_sold_list:
            row = sold + [None] * (max_sold - len(sold))
            sold_data.append(row)
        
        self.countries_bought = pd.DataFrame(
            bought_data,
            index=dates,
            columns=[f'Bought_{i}' for i in range(max_bought)]
        )
        
        self.countries_sold = pd.DataFrame(
            sold_data,
            index=dates,
            columns=[f'Sold_{i}' for i in range(max_sold)]
        )
        
        # Store detailed trade information
        self.historical_trades = trade_details_list
    
    def get_daily_returns(self):
        """
        Calculate daily portfolio returns by upsampling weights to daily frequency.
        
        Methodology:
        1. Map the stored weights (indexed at period-end d_ret) back to their 
           selection date (d_sel).
        2. Forward fill (ffill) these weights to cover all days in the holding period.
        3. Shift by 1 day so that weights determined at Close(t) apply to Returns(t+1).
        """
        if self.historical_weights is None:
            raise ValueError("Backtest has not been run yet. Call run_backtest() first.")

        # 1. Calculate daily returns for the universe
        daily_asset_returns = self.prices.pct_change()

        # 2. Create a DataFrame for weights indexed by SELECTION date (d_sel)
        # Currently self.historical_weights is indexed by d_ret (end of period)
        
        # Create mapping dictionaries
        sel_dates = [t[0] for t in self.date_tuples]  # d_sel
        ret_dates = [t[1] for t in self.date_tuples]  # d_ret
        
        # Initialize empty weights with daily index
        daily_weights = pd.DataFrame(np.nan, index=daily_asset_returns.index, columns=self.historical_weights.columns)
        
        # Map weights to their start dates
        for d_sel, d_ret in zip(sel_dates, ret_dates):
            if d_ret in self.historical_weights.index:
                # Place the weights at the start of the period
                daily_weights.loc[d_sel] = self.historical_weights.loc[d_ret]

        # 3. Forward fill weights
        # This holds the position from d_sel until the next d_sel
        daily_weights = daily_weights.ffill()
        
        # 4. Shift weights by 1 day
        # Essential: Weights decided at Close of d_sel affect returns starting d_sel+1
        daily_weights = daily_weights.shift(1)
        
        # 5. Filter to valid window (start from first selection date)
        if not sel_dates:
             return pd.Series(), pd.Series()
             
        start_date = sel_dates[0]
        daily_weights = daily_weights.loc[start_date:]
        daily_asset_returns = daily_asset_returns.loc[start_date:]
        
        # 6. Calculate Portfolio Return
        daily_port_return = (daily_weights * daily_asset_returns).sum(axis=1)
        
        # 7. Calculate Benchmark Return
        daily_bmk_return = daily_asset_returns[self.bmk] if self.bmk in daily_asset_returns.columns else pd.Series(0, index=daily_asset_returns.index)

        return daily_port_return, daily_bmk_return

    def get_results(self):
        """
        Get all backtest results.
        
        Returns
        -------
        dict
            Dictionary containing all backtest results
        """
        if self.returns is None:
            raise ValueError("Backtest has not been run yet. Call run_backtest() first.")
        
        return {
            'returns': self.returns,
            'returns_net': self.returns_net,
            'benchmark_returns': self.benchmark_returns,
            'active_return': self.active_return,
            'daily_returns': self.daily_returns,
            'daily_benchmark_returns': self.daily_benchmark_returns,
            'historical_countries': self.historical_countries,
            'historical_active_weights': self.historical_active_weights,
            'historical_weights': self.historical_weights,
            'turnover': self.turnover,
            'transaction_costs': self.transaction_costs,
            'countries_bought': self.countries_bought,
            'countries_sold': self.countries_sold,
            'dates': self.dates,
            'date_tuples': self.date_tuples
        }
    
    def get_performance_summary(self, daily=True):
        """
        Calculate and return performance summary statistics.
        
        Uses geometric returns for annualization and adjusts for actual periodicity.
        
        Parameters
        ----------
        daily : bool, optional
            If True, use daily returns for calculation (default True).
            If False, use period-based returns (based on rebalancing frequency).
        
        Returns
        -------
        pd.DataFrame
            DataFrame with performance metrics for portfolio, benchmark, and active
        """
        if self.returns is None:
            raise ValueError("Backtest has not been run yet. Call run_backtest() first.")
        
        # Determine which returns to use
        if daily and self.daily_returns is not None and not self.daily_returns.empty:
            returns_to_use = self.daily_returns
            bmk_returns_to_use = self.daily_benchmark_returns
            periods_per_year = 252
            # Align active returns
            active_returns_to_use = returns_to_use - bmk_returns_to_use
        else:
            # Fallback to period returns if daily requested but not available, or if daily=False
            if daily and (self.daily_returns is None or self.daily_returns.empty):
                print("Warning: Daily returns not available. Using period returns instead.")
            
            returns_to_use = self.returns
            bmk_returns_to_use = self.benchmark_returns
            active_returns_to_use = self.active_return
            # Calculate periods per year based on actual periodicity
            periods_per_year = 252 / self.periodicity
        
        # Calculate number of days in sample
        num_days = (returns_to_use.index[-1] - returns_to_use.index[0]).days
        
        # Calculate cumulative returns
        cum_portfolio = (1 + returns_to_use).cumprod() - 1
        cum_benchmark = (1 + bmk_returns_to_use).cumprod() - 1
        cum_active = cum_portfolio - cum_benchmark
        
        # Helper function to calculate annualized return (geometric)
        def calc_annualized_return(returns_series, days):
            """Calculate annualized return using geometric method."""
            if len(returns_series) == 0 or days <= 0:
                return np.nan
            total_return = (1 + returns_series).prod()
            if total_return <= 0:
                return np.nan
            return total_return ** (365.0 / days) - 1
        
        # Helper function to calculate annualized volatility
        def calc_annualized_vol(returns_series, periods_py):
            """Calculate annualized volatility."""
            if len(returns_series) == 0:
                return np.nan
            return returns_series.std() * np.sqrt(periods_py)
        
        # Helper function to calculate Sharpe Ratio
        def calc_sharpe(ann_return, ann_vol):
            """Calculate Sharpe ratio from annualized metrics."""
            if pd.isna(ann_return) or pd.isna(ann_vol) or ann_vol == 0:
                return np.nan
            return ann_return / ann_vol
        
        # Helper function to calculate Beta
        def calc_beta(portfolio_returns, benchmark_returns):
            """Calculate portfolio beta relative to benchmark."""
            if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
                return np.nan
            # Align the series
            aligned = pd.DataFrame({'port': portfolio_returns, 'bmk': benchmark_returns}).dropna()
            if len(aligned) < 2:
                return np.nan
            cov = aligned['port'].cov(aligned['bmk'])
            var = aligned['bmk'].var()
            if var == 0:
                return np.nan
            return cov / var
        
        # Helper function to calculate Tracking Error
        def calc_tracking_error(active_returns, periods_py):
            """Calculate annualized tracking error."""
            if len(active_returns) == 0:
                return np.nan
            return active_returns.std() * np.sqrt(periods_py)
        
        # Helper function to calculate Up/Down Capture
        def calc_capture_ratios(portfolio_returns, benchmark_returns):
            """Calculate up and down capture ratios."""
            aligned = pd.DataFrame({'port': portfolio_returns, 'bmk': benchmark_returns}).dropna()
            if len(aligned) == 0:
                return np.nan, np.nan
            
            # Up capture: average portfolio return when benchmark is positive / average positive benchmark return
            up_periods = aligned[aligned['bmk'] > 0]
            if len(up_periods) > 0 and up_periods['bmk'].mean() != 0:
                up_capture = up_periods['port'].mean() / up_periods['bmk'].mean()
            else:
                up_capture = np.nan
            
            # Down capture: average portfolio return when benchmark is negative / average negative benchmark return
            down_periods = aligned[aligned['bmk'] < 0]
            if len(down_periods) > 0 and down_periods['bmk'].mean() != 0:
                down_capture = down_periods['port'].mean() / down_periods['bmk'].mean()
            else:
                down_capture = np.nan
            
            return up_capture, down_capture
        
        # Calculate metrics for Portfolio
        port_ann_return = calc_annualized_return(returns_to_use, num_days)
        port_ann_vol = calc_annualized_vol(returns_to_use, periods_per_year)
        port_sharpe = calc_sharpe(port_ann_return, port_ann_vol)
        port_beta = calc_beta(returns_to_use, bmk_returns_to_use)
        port_up_capture, port_down_capture = calc_capture_ratios(returns_to_use, bmk_returns_to_use)
        
        # Calculate metrics for Benchmark
        bmk_ann_return = calc_annualized_return(bmk_returns_to_use, num_days)
        bmk_ann_vol = calc_annualized_vol(bmk_returns_to_use, periods_per_year)
        bmk_sharpe = calc_sharpe(bmk_ann_return, bmk_ann_vol)
        
        # Calculate metrics for Active
        active_ann_return = port_ann_return - bmk_ann_return
        tracking_error = calc_tracking_error(active_returns_to_use, periods_per_year)
        active_info_ratio = calc_sharpe(active_ann_return, tracking_error)
        
        # Build statistics dictionary
        stats = {
            'Portfolio': {
                'Total Return': cum_portfolio.iloc[-1] if len(cum_portfolio) > 0 else np.nan,
                'Annualized Return': port_ann_return,
                'Annualized Volatility': port_ann_vol,
                'Sharpe Ratio': port_sharpe,
                'Max Drawdown': self._calculate_max_drawdown(returns_to_use),
                'Win Rate': (returns_to_use > 0).sum() / len(returns_to_use) if len(returns_to_use) > 0 else np.nan,
                'Beta': port_beta,
                'Up Capture': port_up_capture,
                'Down Capture': port_down_capture,
                'Tracking Error': np.nan
            },
            'Benchmark': {
                'Total Return': cum_benchmark.iloc[-1] if len(cum_benchmark) > 0 else np.nan,
                'Annualized Return': bmk_ann_return,
                'Annualized Volatility': bmk_ann_vol,
                'Sharpe Ratio': bmk_sharpe,
                'Max Drawdown': self._calculate_max_drawdown(bmk_returns_to_use),
                'Win Rate': (bmk_returns_to_use > 0).sum() / len(bmk_returns_to_use) if len(bmk_returns_to_use) > 0 else np.nan,
                'Beta': np.nan,
                'Up Capture': np.nan,
                'Down Capture': np.nan,
                'Tracking Error': np.nan
            },
            'Active': {
                'Total Return': cum_active.iloc[-1] if len(cum_active) > 0 else np.nan,
                'Annualized Return': active_ann_return,
                'Annualized Volatility': tracking_error,
                'Sharpe Ratio': np.nan,
                'Max Drawdown': self._calculate_max_drawdown(active_returns_to_use),
                'Win Rate': (active_returns_to_use > 0).sum() / len(active_returns_to_use) if len(active_returns_to_use) > 0 else np.nan,
                'Beta': np.nan,
                'Up Capture': np.nan,
                'Down Capture': np.nan,
                'Tracking Error': tracking_error,
                'Information Ratio': active_info_ratio
            }
        }
        
        return pd.DataFrame(stats).T
    
    def _calculate_max_drawdown(self, returns):
        """
        Calculate maximum drawdown from a return series.
        
        Parameters
        ----------
        returns : pd.Series
            Series of returns
        
        Returns
        -------
        float
            Maximum drawdown (as a negative number)
        """
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()
    
    def portfolio_turnover_analysis(self, plot=True):
        """
        Perform comprehensive turnover analysis.
        
        Analyzes:
        - Turnover statistics over time
        - Transaction costs impact
        - Countries bought and sold each period
        - Most frequently traded countries
        - Turnover distribution
        
        Parameters
        ----------
        plot : bool, optional
            Whether to generate visualizations (default True)
        
        Returns
        -------
        dict
            Dictionary containing turnover analysis results
        """
        if self.turnover is None:
            raise ValueError("Backtest has not been run yet. Call run_backtest() first.")
        
        # Calculate turnover statistics
        avg_turnover = self.turnover.mean()
        median_turnover = self.turnover.median()
        max_turnover = self.turnover.max()
        min_turnover = self.turnover.min()
        std_turnover = self.turnover.std()
        
        # Calculate transaction cost statistics
        avg_tc = self.transaction_costs.mean()
        total_tc = self.transaction_costs.sum()
        tc_as_pct_of_return = (total_tc / self.returns.sum()) * 100 if self.returns.sum() != 0 else np.nan
        
        # Count countries bought and sold per period
        num_bought = self.countries_bought.notna().sum(axis=1)
        num_sold = self.countries_sold.notna().sum(axis=1)
        
        # Most frequently traded countries
        all_bought = []
        all_sold = []
        for idx in range(len(self.countries_bought)):
            bought_row = self.countries_bought.iloc[idx].dropna().tolist()
            sold_row = self.countries_sold.iloc[idx].dropna().tolist()
            all_bought.extend(bought_row)
            all_sold.extend(sold_row)
        
        # Count trading frequency
        from collections import Counter
        bought_counter = Counter(all_bought)
        sold_counter = Counter(all_sold)
        
        # Combine for total trades
        all_trades = all_bought + all_sold
        total_trades_counter = Counter(all_trades)
        
        # Get top traded countries
        top_n = 10
        most_bought = bought_counter.most_common(top_n)
        most_sold = sold_counter.most_common(top_n)
        most_traded = total_trades_counter.most_common(top_n)
        
        # Calculate gross vs net returns comparison
        cum_gross = (1 + self.returns).cumprod() - 1
        cum_net = (1 + self.returns_net).cumprod() - 1
        gross_final = cum_gross.iloc[-1]
        net_final = cum_net.iloc[-1]
        tc_drag = gross_final - net_final
        
        # (Prints removed for cleanliness)
        
        # Generate visualizations if requested
        if plot:
            self._plot_turnover_analysis(
                num_bought, num_sold, most_traded, 
                bought_counter, sold_counter, total_trades_counter
            )
        
        # Return results dictionary
        self.turnover_analysis_results = {
            'turnover_stats': {
                'mean': avg_turnover,
                'median': median_turnover,
                'max': max_turnover,
                'min': min_turnover,
                'std': std_turnover
            },
            'transaction_cost_stats': {
                'mean': avg_tc,
                'total': total_tc,
                'pct_of_returns': tc_as_pct_of_return,
                'gross_return': gross_final,
                'net_return': net_final,
                'return_drag': tc_drag
            },
            'trading_activity': {
                'avg_bought': num_bought.mean(),
                'avg_sold': num_sold.mean(),
                'unique_countries': len(total_trades_counter),
                'total_events': len(all_trades)
            },
            'most_bought': most_bought,
            'most_sold': most_sold,
            'most_traded': most_traded,
            'country_trade_counts': total_trades_counter
        }
        
        return self.turnover_analysis_results
    
    def _plot_turnover_analysis(self, num_bought, num_sold, most_traded, 
                                 bought_counter, sold_counter, total_trades_counter):
        """
        Generate comprehensive turnover visualizations.
        
        Parameters
        ----------
        num_bought : pd.Series
            Number of countries bought per period
        num_sold : pd.Series
            Number of countries sold per period
        most_traded : list
            List of (country, count) tuples for most traded
        bought_counter : Counter
            Counter of bought countries
        sold_counter : Counter
            Counter of sold countries
        total_trades_counter : Counter
            Counter of all trades
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Turnover over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.turnover.index, self.turnover.values, linewidth=1.5, alpha=0.7, color='steelblue')
        ax1.axhline(y=self.turnover.mean(), color='red', linestyle='--', label=f'Mean: {self.turnover.mean():.2%}', linewidth=2)
        ax1.set_title('Portfolio Turnover Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Turnover')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Plot 2: Transaction costs over time
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.transaction_costs.index, self.transaction_costs.values * 10000, 
                linewidth=1.5, alpha=0.7, color='darkred')
        ax2.set_title('Transaction Costs Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Transaction Cost (bps)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Countries bought and sold per period
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(num_bought.index, num_bought.values, label='Bought', linewidth=1.5, color='green', alpha=0.7)
        ax3.plot(num_sold.index, num_sold.values, label='Sold', linewidth=1.5, color='red', alpha=0.7)
        ax3.set_title('Countries Traded per Period', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Countries')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Turnover distribution
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(self.turnover.values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax4.axvline(x=self.turnover.mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.turnover.mean():.2%}', linewidth=2)
        ax4.set_title('Turnover Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Turnover')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        
        # Plot 5: Most traded countries (total)
        ax5 = fig.add_subplot(gs[2, 0])
        countries_total = [item[0] for item in most_traded[:10]]
        counts_total = [item[1] for item in most_traded[:10]]
        bars = ax5.barh(range(len(countries_total)), counts_total, color='purple', alpha=0.7)
        ax5.set_yticks(range(len(countries_total)))
        ax5.set_yticklabels(countries_total)
        ax5.set_xlabel('Number of Trades')
        ax5.set_title('Most Traded Countries', fontsize=12, fontweight='bold')
        ax5.invert_yaxis()
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Plot 6: Gross vs Net cumulative returns
        ax6 = fig.add_subplot(gs[2, 1])
        cum_gross = (1 + self.returns).cumprod()
        cum_net = (1 + self.returns_net).cumprod()
        ax6.plot(cum_gross.index, cum_gross.values, label='Gross Returns', linewidth=2, color='green')
        ax6.plot(cum_net.index, cum_net.values, label='Net Returns', linewidth=2, color='darkgreen', linestyle='--')
        ax6.set_title('Cumulative Returns: Gross vs Net', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Cumulative Return')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Transaction cost impact (cumulative)
        ax7 = fig.add_subplot(gs[2, 2])
        cum_tc = self.transaction_costs.cumsum()
        ax7.fill_between(cum_tc.index, 0, cum_tc.values * 100, alpha=0.5, color='darkred')
        ax7.plot(cum_tc.index, cum_tc.values * 100, linewidth=2, color='darkred')
        ax7.set_title('Cumulative Transaction Costs', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Date')
        ax7.set_ylabel('Cumulative TC (%)')
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle('Portfolio Turnover Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        if self.save_plots:
            import os
            filepath = os.path.join(self.output_folder, 'turnover_analysis.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  ✓ Plot saved to: {filepath}")
        
        plt.show()
    
    def performance_attribution_analysis(self, plot=True):
        """
        Perform comprehensive performance attribution analysis.
        
        Analyzes the sources of portfolio returns using holding-based attribution:
        - Return contribution by country
        - Active contribution vs benchmark
        - Top/bottom contributors
        - Period-by-period attribution
        - Cumulative attribution over time
        
        Follows industry best practices for performance attribution.
        
        Parameters
        ----------
        plot : bool, optional
            Whether to generate visualizations (default True)
        
        Returns
        -------
        dict
            Dictionary containing attribution analysis results
        """
        if self.returns is None:
            raise ValueError("Backtest has not been run yet. Call run_backtest() first.")
        
        # Step 1: Calculate period returns for each country
        period_country_returns = {}
        
        for idx, (d_sel, d_ret) in enumerate(self.date_tuples):
            period_returns = self._calculate_period_returns(d_sel, d_ret)
            period_country_returns[d_ret] = period_returns
        
        # Convert to DataFrame
        returns_df = pd.DataFrame(period_country_returns).T
        returns_df = returns_df.fillna(0)
        
        # Step 2: Calculate return contribution for each country in each period
        # Contribution = Weight Ã— Return
        contribution_df = pd.DataFrame(index=self.historical_weights.index)
        
        for country in self.historical_weights.columns:
            if country in returns_df.columns:
                contribution_df[country] = self.historical_weights[country] * returns_df[country]
            else:
                contribution_df[country] = 0.0
        
        # Step 3: Calculate active weights (portfolio weight - benchmark weight)
        # For this strategy, benchmark weight is either explicit or 0 for countries
        active_weights_df = self.historical_weights.copy()
        if self.bmk in active_weights_df.columns:
            # All non-benchmark positions are active
            active_weights_df = active_weights_df.drop(columns=[self.bmk])
        
        # Step 4: Calculate active contribution
        # Active Contribution = Weight Ã— (Country Return - Benchmark Return)
        active_contribution_df = pd.DataFrame(index=self.historical_weights.index)
        
        for country in active_weights_df.columns:
            if country in returns_df.columns:
                country_active_return = returns_df[country] - self.benchmark_returns
                active_contribution_df[country] = active_weights_df[country] * country_active_return
            else:
                active_contribution_df[country] = 0.0
        
        # Step 5: Calculate cumulative contributions
        cumulative_contribution = contribution_df.sum(axis=0).sort_values(ascending=False)
        cumulative_active_contribution = active_contribution_df.sum(axis=0).sort_values(ascending=False)
        
        # Step 6: Identify top and bottom contributors
        top_n = 10
        top_contributors = cumulative_contribution.head(top_n)
        bottom_contributors = cumulative_contribution.tail(top_n)
        
        top_active_contributors = cumulative_active_contribution.head(top_n)
        bottom_active_contributors = cumulative_active_contribution.tail(top_n)
        
        # Step 7: Calculate statistics
        total_portfolio_return = self.returns.sum()
        total_active_return = self.active_return.sum()
        total_benchmark_return = self.benchmark_returns.sum()
        
        # Attribution accuracy check
        sum_contributions = contribution_df.sum(axis=1).sum()
        attribution_diff = total_portfolio_return - sum_contributions
        
        # Country participation statistics
        country_participation = (self.historical_weights > 0).sum(axis=0).sort_values(ascending=False)
        country_participation = country_participation[country_participation.index != self.bmk]
        
        avg_weight_by_country = self.historical_weights.mean(axis=0).sort_values(ascending=False)
        avg_weight_by_country = avg_weight_by_country[avg_weight_by_country.index != self.bmk]
        
        # Step 8: Period analysis - best/worst performing periods
        period_portfolio_returns = self.returns.copy()
        best_periods = period_portfolio_returns.nlargest(5)
        worst_periods = period_portfolio_returns.nsmallest(5)
        
        # Step 9: Calculate hit rate by country (% of positive contributions)
        country_hit_rates = {}
        for country in contribution_df.columns:
            if country != self.bmk:
                positive_periods = (contribution_df[country] > 0).sum()
                total_periods_held = (self.historical_weights[country] > 0).sum()
                if total_periods_held > 0:
                    country_hit_rates[country] = positive_periods / total_periods_held
        
        hit_rate_series = pd.Series(country_hit_rates).sort_values(ascending=False)
        
        # Generate visualizations if requested
        if plot:
            self._plot_attribution_analysis(
                contribution_df, active_contribution_df,
                cumulative_contribution, cumulative_active_contribution,
                country_participation, avg_weight_by_country, hit_rate_series
            )
        
        # Return comprehensive results
        self.attribution_analysis_results = {
            'contribution_by_period': contribution_df,
            'active_contribution_by_period': active_contribution_df,
            'cumulative_contribution': cumulative_contribution,
            'cumulative_active_contribution': cumulative_active_contribution,
            'top_contributors': top_contributors,
            'bottom_contributors': bottom_contributors,
            'top_active_contributors': top_active_contributors,
            'bottom_active_contributors': bottom_active_contributors,
            'country_participation': country_participation,
            'avg_weight_by_country': avg_weight_by_country,
            'hit_rates': hit_rate_series,
            'total_return': total_portfolio_return,
            'total_active_return': total_active_return,
            'attribution_check': attribution_diff,
            'best_periods': best_periods,
            'worst_periods': worst_periods
        }
        
        return self.attribution_analysis_results
    
    def _plot_attribution_analysis(self, contribution_df, active_contribution_df,
                                    cumulative_contribution, cumulative_active_contribution,
                                    country_participation, avg_weight_by_country, hit_rate_series):
        """
        Generate comprehensive attribution visualizations.
        
        Parameters
        ----------
        contribution_df : pd.DataFrame
            Return contributions by country and period
        active_contribution_df : pd.DataFrame
            Active contributions by country and period
        cumulative_contribution : pd.Series
            Cumulative contribution by country
        cumulative_active_contribution : pd.Series
            Cumulative active contribution by country
        country_participation : pd.Series
            Number of periods each country was held
        avg_weight_by_country : pd.Series
            Average weight by country
        hit_rate_series : pd.Series
            Hit rate by country
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        # Plot 1: Top 10 Cumulative Contributors
        ax1 = fig.add_subplot(gs[0, 0])
        top_10 = cumulative_contribution.head(10)
        colors_top = ['green' if x > 0 else 'red' for x in top_10.values]
        bars = ax1.barh(range(len(top_10)), top_10.values, color=colors_top, alpha=0.7)
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels(top_10.index, fontsize=9)
        ax1.set_xlabel('Cumulative Contribution')
        ax1.set_title('Top 10 Contributors (Total Return)', fontsize=11, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
        
        # Plot 2: Bottom 10 Contributors
        ax2 = fig.add_subplot(gs[0, 1])
        bottom_10 = cumulative_contribution.tail(10).sort_values()
        colors_bottom = ['green' if x > 0 else 'red' for x in bottom_10.values]
        ax2.barh(range(len(bottom_10)), bottom_10.values, color=colors_bottom, alpha=0.7)
        ax2.set_yticks(range(len(bottom_10)))
        ax2.set_yticklabels(bottom_10.index, fontsize=9)
        ax2.set_xlabel('Cumulative Contribution')
        ax2.set_title('Bottom 10 Contributors (Total Return)', fontsize=11, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
        
        # Plot 3: Top 10 Active Contributors
        ax3 = fig.add_subplot(gs[0, 2])
        top_10_active = cumulative_active_contribution.head(10)
        colors_active = ['green' if x > 0 else 'red' for x in top_10_active.values]
        ax3.barh(range(len(top_10_active)), top_10_active.values, color=colors_active, alpha=0.7)
        ax3.set_yticks(range(len(top_10_active)))
        ax3.set_yticklabels(top_10_active.index, fontsize=9)
        ax3.set_xlabel('Active Contribution')
        ax3.set_title('Top 10 Active Contributors (vs Benchmark)', fontsize=11, fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
        
        # Plot 4: Cumulative Attribution Over Time (Top 5)
        ax4 = fig.add_subplot(gs[1, :])
        top_5_countries = cumulative_contribution.head(5).index
        for country in top_5_countries:
            if country in contribution_df.columns:
                cum_contrib = contribution_df[country].cumsum()
                ax4.plot(cum_contrib.index, cum_contrib.values, label=country, linewidth=2, alpha=0.8)
        ax4.set_title('Cumulative Attribution Over Time (Top 5 Contributors)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Contribution')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Plot 5: Period-by-Period Attribution (Stacked Area - Top 5)
        ax5 = fig.add_subplot(gs[2, :])
        top_5_df = contribution_df[top_5_countries].copy()
        # Fill any missing dates
        top_5_df = top_5_df.fillna(0)
        ax5.stackplot(top_5_df.index, *[top_5_df[col].values for col in top_5_df.columns],
                     labels=top_5_df.columns, alpha=0.7)
        ax5.set_title('Period-by-Period Attribution (Top 5 Contributors - Stacked)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Period Contribution')
        ax5.legend(loc='upper left', fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Country Participation (Periods Held)
        ax6 = fig.add_subplot(gs[3, 0])
        top_participation = country_participation.head(15)
        ax6.barh(range(len(top_participation)), top_participation.values, color='steelblue', alpha=0.7)
        ax6.set_yticks(range(len(top_participation)))
        ax6.set_yticklabels(top_participation.index, fontsize=8)
        ax6.set_xlabel('Number of Periods')
        ax6.set_title('Country Participation (Periods Held)', fontsize=11, fontweight='bold')
        ax6.invert_yaxis()
        ax6.grid(True, alpha=0.3, axis='x')
        
        # Plot 7: Average Weight by Country
        ax7 = fig.add_subplot(gs[3, 1])
        top_weights = avg_weight_by_country.head(15)
        ax7.barh(range(len(top_weights)), top_weights.values, color='purple', alpha=0.7)
        ax7.set_yticks(range(len(top_weights)))
        ax7.set_yticklabels(top_weights.index, fontsize=8)
        ax7.set_xlabel('Average Weight')
        ax7.set_title('Average Portfolio Weight by Country', fontsize=11, fontweight='bold')
        ax7.invert_yaxis()
        ax7.grid(True, alpha=0.3, axis='x')
        ax7.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
        
        # Plot 8: Hit Rate by Country
        ax8 = fig.add_subplot(gs[3, 2])
        top_hit_rates = hit_rate_series.head(15)
        colors_hit = ['green' if x >= 0.5 else 'orange' for x in top_hit_rates.values]
        ax8.barh(range(len(top_hit_rates)), top_hit_rates.values, color=colors_hit, alpha=0.7)
        ax8.set_yticks(range(len(top_hit_rates)))
        ax8.set_yticklabels(top_hit_rates.index, fontsize=8)
        ax8.set_xlabel('Hit Rate (% Positive)')
        ax8.set_title('Hit Rate by Country (% Positive Contributions)', fontsize=11, fontweight='bold')
        ax8.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax8.invert_yaxis()
        ax8.grid(True, alpha=0.3, axis='x')
        ax8.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        
        plt.suptitle('Performance Attribution Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        if self.save_plots:
            import os
            filepath = os.path.join(self.output_folder, 'attribution_analysis.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  ✓ Plot saved to: {filepath}")
        
        plt.show()
    
    def IC_analysis(self, rolling_window=20, plot=True):
        """
        Perform Information Coefficient (IC) analysis.
        
        Implements the standardized IC analysis methodology consistent with test_normalized_scores.py,
        adapted for the Backtest class variables.
        
        Parameters
        ----------
        rolling_window : int, optional
            Window size for rolling IC calculation (default 20 periods)
        plot : bool, optional
            Whether to generate visualizations (default True)
        
        Returns
        -------
        dict
            Dictionary containing IC analysis results
        """
        if self.returns is None:
            raise ValueError("Backtest has not been run yet. Call run_backtest() first.")
        
        # ------------------------------------------------------------------------
        # 1. Setup Variables
        # ------------------------------------------------------------------------
        periodicity = self.periodicity
        method = self.selection_criteria
        
        # Extract dates used in the backtest to ensure consistency
        # We use the same logic as test_normalized_scores.py: 
        # Start from the most recent date and step backwards by periodicity
        # User preference: Use normalized_score index for date determination
        all_dates = self.normalized_score.index
        selected_dates = sorted(all_dates[::-periodicity])
        
        # Filter selected_dates to ensure they exist in normalized_score (redundant but safe)
        # and prices (to ensure we can calculate returns)
        valid_dates = self.prices.index.intersection(selected_dates)
        selected_dates = sorted(valid_dates)
        
        if len(selected_dates) < 2:
            # Warning is useful here as it indicates a data issue
            print("Warning: Not enough valid dates for IC analysis.")
            return {}
        
        # ------------------------------------------------------------------------
        # 2. Calculate Period Returns
        # ------------------------------------------------------------------------
        # Calculate returns between consecutive rebalancing dates
        # Returns[t1] = (Price[t1] - Price[t0]) / Price[t0]
        try:
            # Ensure all data is numeric
            prices_numeric = self.prices.apply(pd.to_numeric, errors='coerce')
            
            # Calculate returns between consecutive rebalancing dates
            period_returns_df = prices_numeric.loc[selected_dates].pct_change()
            period_returns_df = period_returns_df.iloc[1:]
            
            # Drop benchmark if present
            period_returns_df = period_returns_df.drop(columns=[self.bmk], errors='ignore')
            
        except Exception as e:
            print(f"\n✗ ERROR calculating period returns:")
            print(f"  {str(e)}")
            raise
            
        # ------------------------------------------------------------------------
        # 3. Calculate Signal
        # ------------------------------------------------------------------------
        if method == 'absolute':
            # ABSOLUTE: Use current scores as signal
            # Signal[t] = Score[t]
            signal_df = self.normalized_score.loc[selected_dates]
            signal_description = "Absolute Scores"
        else:  # relative
            # RELATIVE: Use score changes over periodicity as signal
            # Signal[t] = Score[t] - Score[t-periodicity]
            signal_df = self.normalized_score.loc[selected_dates].diff().iloc[1:]
            signal_description = "Score Changes (Relative)"
            
        # ------------------------------------------------------------------------
        # 4. Calculate IC Statistics
        # ------------------------------------------------------------------------
        ic_values = []
        ic_dates = []
        ic_details = []
        
        from scipy.stats import spearmanr
        
        # Align dates and create (start, end) pairs for each period
        common_dates = signal_df.index.intersection(period_returns_df.index)
        dates = [(common_dates[i], common_dates[i+1]) for i in range(len(common_dates)-1)]
        
        for date in dates:
            # Get signal at period start (t0) and return at period end (t1)
            # Note: period_returns_df[t1] represents the return from t0 -> t1
            t0 = date[0]
            t1 = date[1]
            
            signal = signal_df.loc[t0]
            period_returns = period_returns_df.loc[t1]
            
            # Get common countries (both have data)
            common_countries = signal.dropna().index.intersection(period_returns.dropna().index)
            
            if len(common_countries) >= 3:  # Need at least 3 points
                signal_values = signal[common_countries].values
                return_values = period_returns[common_countries].values
                    
                # Calculate Spearman rank correlation
                ic, p_value = spearmanr(signal_values, return_values)
                
                ic_values.append(ic)
                ic_dates.append(t0)
                ic_details.append({
                    'date': t0,
                    'ic': ic,
                    'p_value': p_value,
                    'num_countries': len(common_countries)
                })
        
        if not ic_values:
            return {}
            
        # Create Series
        ic_series = pd.Series(ic_values, index=ic_dates, name='IC')
        
        # ------------------------------------------------------------------------
        # 5. Calculate Aggregate Statistics
        # ------------------------------------------------------------------------
        mean_ic = ic_series.mean()
        median_ic = ic_series.median()
        std_ic = ic_series.std()
        n_periods = len(ic_series)
        
        # IC t-statistic
        ic_t_stat = mean_ic / (std_ic / np.sqrt(n_periods)) if std_ic != 0 else 0
        
        # ICIR (IC Information Ratio)
        icir = mean_ic / std_ic if std_ic != 0 else 0
        
        # Hit rate
        hit_rate = (ic_series > 0).sum() / len(ic_series) if len(ic_series) > 0 else 0
        
        # Rolling stats
        rolling_ic = ic_series.rolling(window=rolling_window, min_periods=rolling_window//2).mean()
        rolling_ic_std = ic_series.rolling(window=rolling_window, min_periods=rolling_window//2).std()
        cumulative_ic = ic_series.cumsum()
        
        # ------------------------------------------------------------------------
        # 6. Store and Return Results
        # ------------------------------------------------------------------------
        
        # Generate visualizations if requested
        if plot:
            self._plot_IC_analysis(ic_series, rolling_ic, rolling_ic_std, 
                                  cumulative_ic, rolling_window)
        
        # Return comprehensive results
        self.ic_analysis_results = {
            'ic_series': ic_series,
            'rolling_ic': rolling_ic,
            'rolling_ic_std': rolling_ic_std,
            'cumulative_ic': cumulative_ic,
            'statistics': {
                'mean_ic': mean_ic,
                'median_ic': median_ic,
                'std_ic': std_ic,
                'ic_t_stat': ic_t_stat,
                'icir': icir,
                'hit_rate': hit_rate,
                'min_ic': ic_series.min(),
                'max_ic': ic_series.max(),
                'num_periods': n_periods
            },
            'ic_details': ic_details,
            'signal_type': signal_description
        }
        
        return self.ic_analysis_results
    
    def _plot_IC_analysis(self, ic_series, rolling_ic, rolling_ic_std, 
                          cumulative_ic, rolling_window):
        """
        Generate comprehensive IC analysis visualizations.
        
        Parameters
        ----------
        ic_series : pd.Series
            IC values by period
        rolling_ic : pd.Series
            Rolling mean IC
        rolling_ic_std : pd.Series
            Rolling standard deviation of IC
        cumulative_ic : pd.Series
            Cumulative IC
        rolling_window : int
            Rolling window size
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: IC Over Time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.bar(ic_series.index, ic_series.values, alpha=0.6, 
               color=['green' if x > 0 else 'red' for x in ic_series.values],
               width=np.diff(ic_series.index.append(pd.Index([ic_series.index[-1]]))).astype('timedelta64[D]').astype(int)[0] * 0.8)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax1.axhline(y=ic_series.mean(), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean IC: {ic_series.mean():.4f}')
        ax1.set_title('Information Coefficient (IC) Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('IC (Spearman Correlation)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling IC
        ax2 = fig.add_subplot(gs[1, 0:2])
        ax2.plot(rolling_ic.index, rolling_ic.values, linewidth=2, color='blue', label=f'{rolling_window}-Period Rolling IC')
        # Add confidence bands
        if rolling_ic_std is not None and not rolling_ic_std.isna().all():
            upper_band = rolling_ic + 1.96 * rolling_ic_std / np.sqrt(rolling_window)
            lower_band = rolling_ic - 1.96 * rolling_ic_std / np.sqrt(rolling_window)
            ax2.fill_between(rolling_ic.index, lower_band, upper_band, alpha=0.2, color='blue')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(y=ic_series.mean(), color='red', linestyle='--', 
                   linewidth=1.5, label=f'Overall Mean: {ic_series.mean():.4f}')
        ax2.set_title(f'Rolling {rolling_window}-Period IC', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Rolling IC')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: IC Distribution
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.hist(ic_series.values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax3.axvline(x=ic_series.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {ic_series.mean():.4f}')
        ax3.axvline(x=ic_series.median(), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {ic_series.median():.4f}')
        ax3.set_title('IC Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('IC')
        ax3.set_ylabel('Frequency')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Cumulative IC
        ax4 = fig.add_subplot(gs[2, 0:2])
        ax4.plot(cumulative_ic.index, cumulative_ic.values, linewidth=2, color='darkgreen')
        ax4.fill_between(cumulative_ic.index, 0, cumulative_ic.values, alpha=0.3, color='green')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_title('Cumulative IC Over Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative IC')
        ax4.grid(True, alpha=0.3)
        
        # Add trend line
        from scipy import stats
        x_numeric = np.arange(len(cumulative_ic))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, cumulative_ic.values)
        trend_line = slope * x_numeric + intercept
        ax4.plot(cumulative_ic.index, trend_line, 'r--', linewidth=2, 
                alpha=0.7, label=f'Trend (slope={slope:.4f})')
        ax4.legend(loc='best')
        
        # Plot 5: IC Statistics Summary (Text Box)
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')
        
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ic_t_stat = mean_ic / (std_ic / np.sqrt(len(ic_series))) if std_ic != 0 else 0
        icir = mean_ic / std_ic if std_ic != 0 else 0
        hit_rate = (ic_series > 0).sum() / len(ic_series)
        
        stats_text = f"""
        IC STATISTICS SUMMARY
        {'=' * 35}
        
        Mean IC:           {mean_ic:>10.4f}
        Median IC:         {ic_series.median():>10.4f}
        Std Dev IC:        {std_ic:>10.4f}
        
        IC t-statistic:    {ic_t_stat:>10.2f}
        ICIR:              {icir:>10.4f}
        Hit Rate:          {hit_rate:>10.2%}
        
        Min IC:            {ic_series.min():>10.4f}
        Max IC:            {ic_series.max():>10.4f}
        
        Periods:           {len(ic_series):>10d}
        
        {'=' * 35}
        Significance:
        {'Significant' if abs(ic_t_stat) > 1.96 else 'Not Significant'}
        
        Quality:
        {'Strong' if abs(mean_ic) > 0.05 else 'Weak'}
        """
        
        ax5.text(0.1, 0.95, stats_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Information Coefficient (IC) Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if self.save_plots:
            import os
            filepath = os.path.join(self.output_folder, 'ic_analysis.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  ✓ Plot saved to: {filepath}")
        
        plt.show()
    
    def plot_cumulative_returns(self):
        """Plot cumulative returns for portfolio, benchmark, and active."""
        if self.returns is None:
            raise ValueError("Backtest has not been run yet. Call run_backtest() first.")
        
        # Use daily returns if available for smoother plots
        if self.daily_returns is not None and not self.daily_returns.empty:
            returns_to_use = self.daily_returns
            bmk_returns_to_use = self.daily_benchmark_returns
            active_returns_to_use = returns_to_use - bmk_returns_to_use
        else:
            returns_to_use = self.returns
            bmk_returns_to_use = self.benchmark_returns
            active_returns_to_use = self.active_return
        
        cum_portfolio = (1 + returns_to_use).cumprod()
        cum_benchmark = (1 + bmk_returns_to_use).cumprod()
        cum_active = cum_portfolio - cum_benchmark
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Portfolio vs Benchmark
        ax1.plot(cum_portfolio.index, cum_portfolio.values, label='Portfolio', linewidth=2)
        ax1.plot(cum_benchmark.index, cum_benchmark.values, label='Benchmark', linewidth=2, alpha=0.7)
        ax1.set_title('Cumulative Returns: Portfolio vs Benchmark', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Active Return
        ax2.plot(cum_active.index, cum_active.values, label='Active Return', color='green', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Cumulative Active Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Active Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            import os
            filepath = os.path.join(self.output_folder, 'cumulative_returns.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  ✓ Plot saved to: {filepath}")
        
        plt.show()
    
    def plot_weights_over_time(self, top_n=None):
        """
        Plot portfolio weights over time as a stacked area chart.
        
        Parameters
        ----------
        top_n : int, optional
            Number of top countries to display individually. Others will be grouped into 'Other'.
            If None (default), all countries are shown (can be messy if many countries).
        """
        if self.historical_weights is None:
            raise ValueError("Backtest has not been run yet. Call run_backtest() first.")
        
        # 1. Get Weights Data
        # We want to show the evolution, so we should ideally use daily weights if available 
        # for a smoother chart, or period weights if not. 
        # To be consistent with the 'holdings' view, let's use the period weights (historical_weights)
        # but forward filled to show the holding periods clearly, OR just plot the rebalancing points.
        # A stacked area chart looks best with a continuous index.
        
        # Let's reconstruct the daily weights logic locally to get a smooth area chart
        daily_weights = self.historical_weights.reindex(self.daily_returns.index, method='ffill') if self.daily_returns is not None else self.historical_weights
        
        # Fill NaNs with 0
        daily_weights = daily_weights.fillna(0)

        # 2. Handle "Other" grouping if top_n is specified
        plot_data = daily_weights.copy()
        
        if top_n is not None and len(plot_data.columns) > top_n:
            # Calculate average weight for ranking
            avg_weights = plot_data.mean()
            
            # Keep Benchmark separate if it exists
            if self.bmk in plot_data.columns:
                avg_weights_no_bmk = avg_weights.drop(self.bmk)
                top_countries = avg_weights_no_bmk.nlargest(top_n).index.tolist()
                countries_to_keep = top_countries + [self.bmk]
            else:
                top_countries = avg_weights.nlargest(top_n).index.tolist()
                countries_to_keep = top_countries
            
            # Group others
            others_mask = ~plot_data.columns.isin(countries_to_keep)
            if others_mask.any():
                plot_data['Other'] = plot_data.loc[:, others_mask].sum(axis=1)
                plot_data = plot_data.loc[:, countries_to_keep + ['Other']]
        
        # 3. Plot Stacked Area Chart
        # Use 'tab20' colormap for distinct colors
        import matplotlib.cm as cm
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create the stackplot
        # Sort columns: Put 'Other' at the bottom, then smallest to largest, or largest to smallest.
        # A good heuristic: Benchmark at bottom (if present), then Other, then largest countries.
        avg_w = plot_data.mean()
        
        # Custom sort order
        sorted_cols = avg_w.sort_values(ascending=True).index.tolist()
        
        # Ensure 'Other' is at the bottom if it exists
        if 'Other' in sorted_cols:
            sorted_cols.remove('Other')
            sorted_cols.insert(0, 'Other')
            
        # Ensure Benchmark is at the very bottom if it exists (foundation)
        if self.bmk in sorted_cols:
            sorted_cols.remove(self.bmk)
            sorted_cols.insert(0, self.bmk)
            
        plot_data = plot_data[sorted_cols]
        
        # Generate colors
        # We use tab20, but if we have more than 20, we cycle or interpolate
        n_cols = len(plot_data.columns)
        if n_cols <= 20:
            colors = cm.tab20.colors[:n_cols]
        else:
            # Interpolate if more than 20 needed (rare with top_n=10)
            colors = [cm.tab20(i) for i in np.linspace(0, 1, n_cols)]
            
        # Draw Stackplot
        stacks = ax.stackplot(plot_data.index, plot_data.T, labels=plot_data.columns, colors=colors, alpha=0.85)
        
        ax.set_title('Portfolio Weights Evolution (Stacked Area)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight')
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.grid(True, alpha=0.3)
        
        # --- FIX LEGEND OVERLAP ---
        # Shrink the axis width by 15% to make room for the legend on the right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        
        # Place legend in the reserved space
        # Reverse legend order to match stack order (visual intuition)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', borderaxespad=0.)
        
        if self.save_plots:
            import os
            filepath = os.path.join(self.output_folder, 'weights_over_time.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  ✓ Plot saved to: {filepath}")
        
        plt.show()
