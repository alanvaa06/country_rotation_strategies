"""
Country Rotation Strategy - Main Script

This script implements a quantitative country rotation strategy using financial and economic data.
It processes multiple datasets, performs regional aggregations, calculates derived metrics,
and exports the results for further analysis.

All functions are imported from function_module.py for better code organization.
The FactorTransformer class is imported from FactorTransformer.py.
"""

import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns

from FactorTransformer import FactorTransformer
from ProcessData import ProcessData

warnings.filterwarnings('ignore')

# Set working directory
#os.chdir('D:/Users/avazquez/OneDrive - valmexcasabolsa/Documents/QuantModels/country_rotation')

# Configure plotting style
plt.style.use('seaborn-v0_8')

#%%
def run_inputs():
    """
    Main execution function for the country rotation strategy.

    This function orchestrates the entire data processing pipeline:
    1. Load raw data from Excel files
    2. Load classification data
    3. Remove weekends from time series
    4. Slice data by target date and drop problematic countries
    5. Create regional classifications
    6. Transform and process all financial metrics
    7. Export processed data
    """

    print("=" * 60)
    print("COUNTRY ROTATION STRATEGY - DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    try:
        # Initialize ProcessData instance
        processor = ProcessData(
            inputs_folder='Inputs',
            classification_file='Classification.xlsx',
            target_date='2010-01-01',
            columns_to_drop=['Saudi Arabia']
        )
        
        # Run full pipeline with export enabled
        processed_data, regions_dict, classification = processor.run_full_pipeline(export_data=False)
        
        # Show some key metrics that were created
        # Get original file names (before derived metrics were added)
        file_names = processor.original_file_names
        derived_metrics = [k for k in processed_data.keys() if k not in file_names]
        if derived_metrics:
            print(f"\n🔧 NEW DERIVED METRICS ({len(derived_metrics)}):")
            for i, metric in enumerate(derived_metrics[:10], 1):  # Show first 10
                print(f"   {i}. {metric}")
            if len(derived_metrics) > 10:
                print(f"   ... and {len(derived_metrics) - 10} more derived metrics")
        
        print(f"\n✅ All processed data exported to: ProcessedInputs/")
        print(f"✅ Ready for quantitative analysis and strategy implementation!")
        
        return processed_data, regions_dict, classification, processor
        
    except Exception as e:
        print(f"\n❌ ERROR IN PIPELINE EXECUTION:")
        print(f"   {str(e)}")
        print(f"\n🔧 Please check your data files and try again.")
        raise


# Execute the main pipeline
processed_data, regions_dict, classification, processor = run_inputs()

# Make key variables available in the global scope for interactive use
dataFrames = processed_data

print(f"\n📋 VARIABLES AVAILABLE FOR ANALYSIS:")
print(f"   • dataFrames: {len(dataFrames)} processed datasets")
print(f"   • regions_dict: Regional country groupings")
print(f"   • classification: Country classification DataFrame")
print(f"   • processed_data: Same as dataFrames (alias)")

# Quick data exploration
processor.explore_data()

#%%

# ============================================================================
# FACTOR TRANSFORMATION AND ANALYSIS
# ============================================================================

# Prepare factor dataframes for transformation
factor_dfs = dataFrames.copy()

# ============================================================================
# FACTOR ANALYSIS CONFIGURATION
# ============================================================================

# Define weights for metric combinations
weights = {
    'zscore': 0.25,
    'absolute_pct': 0.25,
    'relative_rank': 0.1,
    'delta_pct': 0.4
}

# Define weights for category combinations
category_weights = {
    'Quality': 0.25,
    'Valuation': 0.1,
    'Profitability': 0.35,
    'Momentum': 0.3
}
#%%
# ============================================================================
# PHASE 1: INITIAL ANALYSIS (Full Factor Set)
# ============================================================================

# Step 1: Initialize FactorTransformer with all factors
factor_transformer = FactorTransformer('World')

# Step 2: Transform all factors
factor_results = factor_transformer.transform_all(factor_dfs)

# Step 3: Calculate weighted averages
weighted_scores = factor_transformer.calculate_weighted_average(factor_results, weights)

category_scores, country_scores = factor_transformer.aggregate_by_category(weighted_scores)

composite_scores, contributions = factor_transformer.calculate_composite_score(category_weights)

normalized_scores, rebased_by_country, rebased_by_factor = factor_transformer.normalize_and_rebase_contributions()

# Step 4: Analyze factor redundancy
factor_redundancy_results = factor_transformer.analyze_factor_redundancy(
    distance_threshold=0.5,
    linkage_method='ward',
    selection_criterion='unique' # or 'unique' or 'central'
)

# Step 5: Visualize clusters
factor_transformer.plot_factor_dendrogram()

# Step 6: Review recommendations
print("\nRecommended factors:")
for factor in factor_redundancy_results['recommended_factors']:
    category = factor_transformer.factor_map[factor]
    print(f"  {factor} ({category}): {factor_redundancy_results['selection_rationale'][factor]}")

#%%
selected_factors=['Debt', 'AssetsEquity', 'DvdYieldTTMSpread',
                  'EbitdaMargin', 'ROE', 'Equity', 'EV', 'EV_EBIT',
                  'Net_Debt_Ebitda', 'Flows', 'Fwd_EV_EBITDA',
                  'CashFlowYieldFWDSpread', 'Fwd_PS', 'Fwd_ROE',
                  'GDP', 'M2', 'CF', 'Revenue', 'SI_Ratio', 'Ten_Year',
                  'ConsensusSalesGrowth', 'ConsensusEbitdaGrowth',
                  'ConsensusEarningsGrowth', 'ConsensusCashFlowGrowth',
                  'EarningsYieldTTMSpread', 'NetMargin', 'FwdRevenue',
                  'FwdEBITDAMargin', 'RollingEarnings', 'FwdRollingEarnings',
                  'RollingVol', 'CumFlow']


#%%
# ============================================================================
# PHASE 2: FILTERED PIPELINE (Selected Factors)
# ============================================================================

# Step 7: Filter raw data to selected factors
selected_factors = factor_redundancy_results['recommended_factors']
filtered_factor_dfs = {
    factor: df for factor, df in factor_dfs.items() 
    if factor in selected_factors
}

# Step 8: Re-initialize FactorTransformer for reduced factor set
factor_transformer_reduced = FactorTransformer('World')

# Step 9: Re-run entire pipeline with filtered factors
factor_results_reduced = factor_transformer_reduced.transform_all(filtered_factor_dfs)

weighted_scores_reduced = factor_transformer_reduced.calculate_weighted_average(
    factor_results_reduced, 
    weights
)

category_scores_reduced, country_scores_reduced = factor_transformer_reduced.aggregate_by_category(
    weighted_scores_reduced
)

composite_scores_reduced, contributions_reduced = factor_transformer_reduced.calculate_composite_score(
    category_weights
)

normalized_scores_reduced, rebased_by_country, rebased_by_factor = factor_transformer_reduced.normalize_and_rebase_contributions()
#%%
factor_changes_reduced, total_changes_reduced = factor_transformer_reduced.calculate_factor_contribution_changes(period=5)


# Step 10: Plot results
factor_transformer_reduced.plot_factor_contributions()

#%%
# Plot normalized scores for each country
for country in normalized_scores.columns:
    normalized_scores[country].plot(title=country.upper(), figsize=(12, 6))
    plt.show()

#%%
for date in factor_transformer_reduced.change_dates[-3:]:
    factor_transformer_reduced.plot_factor_contributions(date)
#%%

dates = factor_transformer_reduced.change_dates
dates_tuple=[(dates[i-1], dates[i]) for i in range(1, len(dates))]
#%%

prices=dataFrames['Price']
returns=prices.diff().dropna()

#%%
class Backtest:
    """
    Backtest class for country rotation strategy.
    
    This class implements a robust backtesting framework that:
    - Selects countries based on absolute or relative scoring criteria
    - Constructs portfolio weights using equal weighting
    - Blends active portfolio with a benchmark
    - Calculates portfolio returns and active returns
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
        Currently supports "Equal" (default)
    bmk : str, optional
        Column name of the Benchmark within the prices DataFrame
    bmk_weight : float, optional
        The fixed weight allocation for the benchmark (default 0.0)
    periodicity : int, optional
        Rebalancing frequency in days (default 5)
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
        periodicity: int = 5
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
        
        # Validate inputs
        self._validate_inputs()
        
        # Extract country columns (exclude benchmark)
        self.countries = [col for col in self.normalized_score.columns if col != self.bmk]
        
        # Initialize result containers
        self.returns = None
        self.historical_countries = None
        self.historical_active_weights = None
        self.historical_weights = None
        self.active_return = None
        self.dates = None
        self.date_tuples = None
        
    def _validate_inputs(self):
        """Validate input parameters and data."""
        # Check selection criteria
        if self.selection_criteria not in ["absolute", "relative"]:
            raise ValueError("selection_criteria must be 'absolute' or 'relative'")
        
        # Check weighting method
        if self.weighting_method not in ["Equal"]:
            raise ValueError("weighting_method currently only supports 'Equal'")
        
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
           - Calculates portfolio returns
           - Calculates active returns
        3. Stores all results as class attributes
        
        Returns
        -------
        dict
            Dictionary containing all backtest results
        """
        print("=" * 60)
        print("RUNNING BACKTEST")
        print("=" * 60)
        print(f"Selection Criteria: {self.selection_criteria.upper()}")
        print(f"Periodicity: {self.periodicity} days")
        print(f"Benchmark Weight: {self.bmk_weight:.1%}")
        print(f"Active Weight: {1 - self.bmk_weight:.1%}")
        
        # Step 1: Filter dates by periodicity
        self.dates = self._filter_dates()
        
        # Step 2: Generate date tuples (selection_date, return_date)
        self.date_tuples = [(self.dates[i-1], self.dates[i]) 
                            for i in range(1, len(self.dates))]
        
        print(f"\nTotal Rebalancing Periods: {len(self.date_tuples)}")
        
        # Initialize storage lists
        portfolio_returns_list = []
        benchmark_returns_list = []
        active_returns_list = []
        selected_countries_list = []
        active_weights_list = []
        blended_weights_list = []
        period_dates = []
        
        # Step 3: Loop through each period
        for idx, (d_sel, d_ret) in enumerate(self.date_tuples):
            # Select countries
            selected_countries = self._select_countries(d_sel)
            
            # Construct weights
            active_weights = self._construct_active_weights(selected_countries)
            blended_weights = self._construct_blended_weights(active_weights)
            
            # Calculate returns for the period
            period_returns = self._calculate_period_returns(d_sel, d_ret)
            
            # Calculate portfolio return
            portfolio_return = self._calculate_portfolio_return(blended_weights, period_returns)
            
            # Calculate benchmark return
            benchmark_return = period_returns.get(self.bmk, 0.0)
            
            # Calculate active return
            active_return = portfolio_return - benchmark_return
            
            # Store results
            portfolio_returns_list.append(portfolio_return)
            benchmark_returns_list.append(benchmark_return)
            active_returns_list.append(active_return)
            selected_countries_list.append(selected_countries)
            active_weights_list.append(active_weights)
            blended_weights_list.append(blended_weights)
            period_dates.append(d_ret)
            
            # Print progress every 50 periods
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(self.date_tuples)} periods...")
        
        # Step 4: Convert results to DataFrames and Series
        self._store_results(
            period_dates,
            portfolio_returns_list,
            benchmark_returns_list,
            active_returns_list,
            selected_countries_list,
            active_weights_list,
            blended_weights_list
        )
        
        print(f"\n✅ Backtest completed successfully!")
        print(f"   Total Periods: {len(self.returns)}")
        print(f"   Average Active Return: {self.active_return.mean():.4%}")
        print(f"   Active Return Std Dev: {self.active_return.std():.4%}")
        
        return self.get_results()
    
    def _filter_dates(self):
        """
        Filter dates from normalized_score using periodicity.
        
        Returns
        -------
        list
            Sorted list of dates at specified periodicity intervals
        """
        # Get all dates from normalized_score
        all_dates = sorted(self.normalized_score.index)
        
        # Filter by periodicity: take every nth date starting from the end
        # Logic: sorted(normalized_score.index[::-period])
        filtered_dates = sorted(all_dates[::-self.periodicity])
        
        return filtered_dates
    
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
    
    def _construct_active_weights(self, selected_countries):
        """
        Construct active portfolio weights using the specified weighting method.
        
        Parameters
        ----------
        selected_countries : list
            List of selected country names
        
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
        else:
            raise ValueError(f"Unsupported weighting method: {self.weighting_method}")
    
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
        benchmark_returns,
        active_returns,
        selected_countries,
        active_weights,
        blended_weights
    ):
        """
        Convert results to pandas objects and store as class attributes.
        
        Parameters
        ----------
        dates : list
            List of period end dates
        portfolio_returns : list
            List of portfolio returns
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
        """
        # Store returns as Series
        self.returns = pd.Series(portfolio_returns, index=dates, name='Portfolio_Return')
        self.benchmark_returns = pd.Series(benchmark_returns, index=dates, name='Benchmark_Return')
        self.active_return = pd.Series(active_returns, index=dates, name='Active_Return')
        
        # Store historical countries as DataFrame
        # Create a DataFrame where each row is a date and columns are country positions (0, 1, 2, ...)
        max_countries = max(len(countries) for countries in selected_countries)
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
            'benchmark_returns': self.benchmark_returns,
            'active_return': self.active_return,
            'historical_countries': self.historical_countries,
            'historical_active_weights': self.historical_active_weights,
            'historical_weights': self.historical_weights,
            'dates': self.dates,
            'date_tuples': self.date_tuples
        }
    
    def get_performance_summary(self):
        """
        Calculate and return performance summary statistics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with performance metrics for portfolio, benchmark, and active
        """
        if self.returns is None:
            raise ValueError("Backtest has not been run yet. Call run_backtest() first.")
        
        # Calculate cumulative returns
        cum_portfolio = (1 + self.returns).cumprod() - 1
        cum_benchmark = (1 + self.benchmark_returns).cumprod() - 1
        cum_active = (1 + self.active_return).cumprod() - 1
        
        # Calculate statistics
        stats = {
            'Portfolio': {
                'Total Return': cum_portfolio.iloc[-1],
                'Annualized Return': self.returns.mean() * 252,
                'Annualized Volatility': self.returns.std() * np.sqrt(252),
                'Sharpe Ratio': (self.returns.mean() / self.returns.std()) * np.sqrt(252) if self.returns.std() != 0 else 0,
                'Max Drawdown': self._calculate_max_drawdown(self.returns),
                'Win Rate': (self.returns > 0).sum() / len(self.returns)
            },
            'Benchmark': {
                'Total Return': cum_benchmark.iloc[-1],
                'Annualized Return': self.benchmark_returns.mean() * 252,
                'Annualized Volatility': self.benchmark_returns.std() * np.sqrt(252),
                'Sharpe Ratio': (self.benchmark_returns.mean() / self.benchmark_returns.std()) * np.sqrt(252) if self.benchmark_returns.std() != 0 else 0,
                'Max Drawdown': self._calculate_max_drawdown(self.benchmark_returns),
                'Win Rate': (self.benchmark_returns > 0).sum() / len(self.benchmark_returns)
            },
            'Active': {
                'Total Return': cum_active.iloc[-1],
                'Annualized Return': self.active_return.mean() * 252,
                'Annualized Volatility': self.active_return.std() * np.sqrt(252),
                'Information Ratio': (self.active_return.mean() / self.active_return.std()) * np.sqrt(252) if self.active_return.std() != 0 else 0,
                'Max Drawdown': self._calculate_max_drawdown(self.active_return),
                'Win Rate': (self.active_return > 0).sum() / len(self.active_return)
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
    
    def plot_cumulative_returns(self):
        """Plot cumulative returns for portfolio, benchmark, and active."""
        if self.returns is None:
            raise ValueError("Backtest has not been run yet. Call run_backtest() first.")
        
        cum_portfolio = (1 + self.returns).cumprod()
        cum_benchmark = (1 + self.benchmark_returns).cumprod()
        cum_active = (1 + self.active_return).cumprod()
        
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
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Cumulative Active Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Active Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_weights_over_time(self, top_n=10):
        """
        Plot portfolio weights over time for top N most frequently selected countries.
        
        Parameters
        ----------
        top_n : int, optional
            Number of top countries to display (default 10)
        """
        if self.historical_weights is None:
            raise ValueError("Backtest has not been run yet. Call run_backtest() first.")
        
        # Get top N countries by average weight (excluding benchmark)
        avg_weights = self.historical_weights.drop(columns=[self.bmk], errors='ignore').mean()
        top_countries = avg_weights.nlargest(top_n).index.tolist()
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot benchmark
        if self.bmk in self.historical_weights.columns:
            ax.plot(self.historical_weights.index, 
                   self.historical_weights[self.bmk], 
                   label=self.bmk, 
                   linewidth=2, 
                   linestyle='--', 
                   color='black', 
                   alpha=0.7)
        
        # Plot top countries
        for country in top_countries:
            ax.plot(self.historical_weights.index, 
                   self.historical_weights[country], 
                   label=country, 
                   alpha=0.7)
        
        ax.set_title(f'Portfolio Weights Over Time (Top {top_n} Countries + Benchmark)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


#%%
# ============================================================================
# BACKTEST EXAMPLE PIPELINE
# ============================================================================

print("\n" + "=" * 80)
print("BACKTEST EXAMPLE PIPELINE - TESTING VARIOUS CONFIGURATIONS")
print("=" * 80)

# Prepare data for backtesting
# Use normalized scores from the reduced factor set
backtest_scores = normalized_scores_reduced.copy()
backtest_prices = prices.copy()  # This should contain the 'Benchmark' column

# Verify that prices has a benchmark column (if not, we'll need to add one)
if 'Benchmark' not in backtest_prices.columns:
    print("\n⚠️  Warning: 'Benchmark' column not found in prices.")
    print("   Creating a synthetic benchmark using equal-weighted portfolio...")
    # Create a simple equal-weighted benchmark
    backtest_prices['Benchmark'] = backtest_prices[normalized_scores_reduced.columns].mean(axis=1)

print(f"\n📊 DATA SUMMARY:")
print(f"   Normalized Scores Shape: {backtest_scores.shape}")
print(f"   Prices Shape: {backtest_prices.shape}")
print(f"   Date Range: {backtest_prices.index[0]} to {backtest_prices.index[-1]}")
print(f"   Countries: {len(backtest_scores.columns)}")

#%%
# ============================================================================
# TEST 1: ABSOLUTE SELECTION WITH NO BENCHMARK WEIGHT
# ============================================================================

print("\n" + "-" * 80)
print("TEST 1: Absolute Selection (Score > 0.75) | 100% Active Portfolio")
print("-" * 80)

backtest_abs = Backtest(
    normalized_score=backtest_scores,
    prices=backtest_prices,
    selection_criteria="absolute",
    absolute_selection_score=0.75,
    weighting_method="Equal",
    bmk="World",
    bmk_weight=0.0,  # 100% active
    periodicity=5
)

# Run backtest
results_abs = backtest_abs.run_backtest()

# Get performance summary
print("\n📈 PERFORMANCE SUMMARY:")
summary_abs = backtest_abs.get_performance_summary()
print(summary_abs.round(4))

# Plot results
print("\n📊 Generating plots...")
backtest_abs.plot_cumulative_returns()
backtest_abs.plot_weights_over_time(top_n=8)

#%%
# ============================================================================
# TEST 2: RELATIVE SELECTION WITH NO BENCHMARK WEIGHT
# ============================================================================

print("\n" + "-" * 80)
print("TEST 2: Relative Selection (Top 5 Score Changes) | 100% Active Portfolio")
print("-" * 80)

backtest_rel = Backtest(
    normalized_score=backtest_scores,
    prices=backtest_prices,
    selection_criteria="relative",
    relative_selection_score=5,
    weighting_method="Equal",
    bmk="Benchmark",
    bmk_weight=0.0,  # 100% active
    periodicity=5
)

# Run backtest
results_rel = backtest_rel.run_backtest()

# Get performance summary
print("\n📈 PERFORMANCE SUMMARY:")
summary_rel = backtest_rel.get_performance_summary()
print(summary_rel.round(4))

# Plot results
print("\n📊 Generating plots...")
backtest_rel.plot_cumulative_returns()
backtest_rel.plot_weights_over_time(top_n=8)

#%%
# ============================================================================
# TEST 3: ABSOLUTE SELECTION WITH 30% BENCHMARK WEIGHT
# ============================================================================

print("\n" + "-" * 80)
print("TEST 3: Absolute Selection (Score > 0.75) | 70% Active / 30% Benchmark")
print("-" * 80)

backtest_blended = Backtest(
    normalized_score=backtest_scores,
    prices=backtest_prices,
    selection_criteria="absolute",
    absolute_selection_score=0.75,
    weighting_method="Equal",
    bmk="Benchmark",
    bmk_weight=0.3,  # 30% benchmark, 70% active
    periodicity=5
)

# Run backtest
results_blended = backtest_blended.run_backtest()

# Get performance summary
print("\n📈 PERFORMANCE SUMMARY:")
summary_blended = backtest_blended.get_performance_summary()
print(summary_blended.round(4))

# Plot results
print("\n📊 Generating plots...")
backtest_blended.plot_cumulative_returns()
backtest_blended.plot_weights_over_time(top_n=8)

#%%
# ============================================================================
# TEST 4: RELATIVE SELECTION WITH 50% BENCHMARK WEIGHT
# ============================================================================

print("\n" + "-" * 80)
print("TEST 4: Relative Selection (Top 5) | 50% Active / 50% Benchmark")
print("-" * 80)

backtest_balanced = Backtest(
    normalized_score=backtest_scores,
    prices=backtest_prices,
    selection_criteria="relative",
    relative_selection_score=5,
    weighting_method="Equal",
    bmk="Benchmark",
    bmk_weight=0.50,  # 50/50 split
    periodicity=5
)

# Run backtest
results_balanced = backtest_balanced.run_backtest()

# Get performance summary
print("\n📈 PERFORMANCE SUMMARY:")
summary_balanced = backtest_balanced.get_performance_summary()
print(summary_balanced.round(4))

# Plot results
print("\n📊 Generating plots...")
backtest_balanced.plot_cumulative_returns()
backtest_balanced.plot_weights_over_time(top_n=8)

#%%
# ============================================================================
# TEST 5: ABSOLUTE SELECTION WITH DIFFERENT THRESHOLD (0.60)
# ============================================================================

print("\n" + "-" * 80)
print("TEST 5: Absolute Selection (Score > 0.60) | 100% Active Portfolio")
print("-" * 80)

backtest_lower_threshold = Backtest(
    normalized_score=backtest_scores,
    prices=backtest_prices,
    selection_criteria="absolute",
    absolute_selection_score=0.60,  # Lower threshold = more countries
    weighting_method="Equal",
    bmk="Benchmark",
    bmk_weight=0.0,
    periodicity=5
)

# Run backtest
results_lower = backtest_lower_threshold.run_backtest()

# Get performance summary
print("\n📈 PERFORMANCE SUMMARY:")
summary_lower = backtest_lower_threshold.get_performance_summary()
print(summary_lower.round(4))

# Plot results
print("\n📊 Generating plots...")
backtest_lower_threshold.plot_cumulative_returns()
backtest_lower_threshold.plot_weights_over_time(top_n=10)

#%%
# ============================================================================
# TEST 6: RELATIVE SELECTION WITH TOP 3 COUNTRIES
# ============================================================================

print("\n" + "-" * 80)
print("TEST 6: Relative Selection (Top 3 Score Changes) | 100% Active Portfolio")
print("-" * 80)

backtest_top3 = Backtest(
    normalized_score=backtest_scores,
    prices=backtest_prices,
    selection_criteria="relative",
    relative_selection_score=3,  # More concentrated
    weighting_method="Equal",
    bmk="Benchmark",
    bmk_weight=0.0,
    periodicity=5
)

# Run backtest
results_top3 = backtest_top3.run_backtest()

# Get performance summary
print("\n📈 PERFORMANCE SUMMARY:")
summary_top3 = backtest_top3.get_performance_summary()
print(summary_top3.round(4))

# Plot results
print("\n📊 Generating plots...")
backtest_top3.plot_cumulative_returns()
backtest_top3.plot_weights_over_time(top_n=5)

#%%
# ============================================================================
# COMPARATIVE ANALYSIS: ALL STRATEGIES
# ============================================================================

print("\n" + "=" * 80)
print("COMPARATIVE ANALYSIS: ALL STRATEGIES")
print("=" * 80)

# Compile all strategies
strategies = {
    'Absolute (0.75) - 100% Active': backtest_abs,
    'Relative (Top 5) - 100% Active': backtest_rel,
    'Absolute (0.75) - 70/30 Blend': backtest_blended,
    'Relative (Top 5) - 50/50 Blend': backtest_balanced,
    'Absolute (0.60) - 100% Active': backtest_lower_threshold,
    'Relative (Top 3) - 100% Active': backtest_top3
}

# Create comparison DataFrame
comparison_data = []
for name, bt in strategies.items():
    summary = bt.get_performance_summary()
    comparison_data.append({
        'Strategy': name,
        'Total Return': summary.loc['Portfolio', 'Total Return'],
        'Ann. Return': summary.loc['Portfolio', 'Annualized Return'],
        'Ann. Volatility': summary.loc['Portfolio', 'Annualized Volatility'],
        'Sharpe Ratio': summary.loc['Portfolio', 'Sharpe Ratio'],
        'Max Drawdown': summary.loc['Portfolio', 'Max Drawdown'],
        'Win Rate': summary.loc['Portfolio', 'Win Rate'],
        'Active Return': summary.loc['Active', 'Total Return'],
        'Info Ratio': summary.loc['Active', 'Information Ratio']
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n📊 STRATEGY COMPARISON:")
print(comparison_df.round(4).to_string(index=False))

# Plot comparative cumulative returns
print("\n📈 Generating comparative plot...")
fig, ax = plt.subplots(figsize=(14, 8))

for name, bt in strategies.items():
    cum_returns = (1 + bt.returns).cumprod()
    ax.plot(cum_returns.index, cum_returns.values, label=name, linewidth=2, alpha=0.8)

# Add benchmark
benchmark_cum = (1 + backtest_abs.benchmark_returns).cumprod()
ax.plot(benchmark_cum.index, benchmark_cum.values, 
        label='Benchmark', linewidth=2.5, linestyle='--', color='black', alpha=0.7)

ax.set_title('Cumulative Returns: All Strategies vs Benchmark', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return', fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# ============================================================================
# DETAILED ANALYSIS: EXAMINE A SPECIFIC BACKTEST
# ============================================================================

print("\n" + "=" * 80)
print("DETAILED ANALYSIS: Examining 'Absolute (0.75) - 100% Active' Strategy")
print("=" * 80)

# Access detailed results
detailed_results = backtest_abs.get_results()

print("\n📋 Available Results:")
for key in detailed_results.keys():
    print(f"   • {key}")

# Show sample of selected countries over time
print("\n🌍 SELECTED COUNTRIES (Last 10 Rebalancing Periods):")
print(backtest_abs.historical_countries.tail(10))

# Show sample of weights over time
print("\n⚖️  PORTFOLIO WEIGHTS (Last 10 Rebalancing Periods):")
print(backtest_abs.historical_weights.tail(10).round(4))

# Calculate and display additional metrics
print("\n📊 ADDITIONAL METRICS:")
print(f"   Average # of Countries Selected: {(backtest_abs.historical_countries.notna().sum(axis=1)).mean():.2f}")
print(f"   Max # of Countries Selected: {(backtest_abs.historical_countries.notna().sum(axis=1)).max():.0f}")
print(f"   Min # of Countries Selected: {(backtest_abs.historical_countries.notna().sum(axis=1)).min():.0f}")

# Calculate turnover (how often weights change)
weight_changes = backtest_abs.historical_weights.diff().abs().sum(axis=1)
print(f"   Average Turnover per Period: {weight_changes.mean():.4f}")

# Best and worst periods
print(f"\n🎯 BEST PERIODS:")
best_periods = backtest_abs.returns.nlargest(5)
for date, ret in best_periods.items():
    print(f"   {date.strftime('%Y-%m-%d')}: {ret:+.2%}")

print(f"\n📉 WORST PERIODS:")
worst_periods = backtest_abs.returns.nsmallest(5)
for date, ret in worst_periods.items():
    print(f"   {date.strftime('%Y-%m-%d')}: {ret:+.2%}")

# Rolling statistics
print("\n📈 ROLLING STATISTICS (252-day windows):")
rolling_sharpe = (backtest_abs.returns.rolling(252).mean() / 
                  backtest_abs.returns.rolling(252).std()) * np.sqrt(252)
print(f"   Average Rolling Sharpe: {rolling_sharpe.mean():.4f}")
print(f"   Max Rolling Sharpe: {rolling_sharpe.max():.4f}")
print(f"   Min Rolling Sharpe: {rolling_sharpe.min():.4f}")

# Plot rolling metrics
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Rolling Sharpe Ratio
ax1.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='blue')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.set_title('Rolling Sharpe Ratio (252-day)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Sharpe Ratio')
ax1.grid(True, alpha=0.3)

# Rolling Active Return
rolling_active = backtest_abs.active_return.rolling(252).sum()
ax2.plot(rolling_active.index, rolling_active.values, linewidth=2, color='green')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_title('Rolling 252-Day Active Return', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Active Return')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("✅ BACKTEST PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)

