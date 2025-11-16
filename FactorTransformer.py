"""
FactorTransformer Module

This module contains the FactorTransformer class for transforming factor data
into standardized metrics for quantitative country rotation strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from typing import Dict


class FactorTransformer:
    """
    Transforms factor data into standardized metrics.
    
    This class provides methods for:
    - Calculating percentile-based metrics (z-scores, absolute/relative rankings)
    - Computing weighted averages across metrics
    - Aggregating factors by category (Valuation, Quality, Profitability, Momentum)
    - Calculating composite scores and contributions
    - Analyzing factor redundancy through hierarchical clustering
    - Visualizing factor contributions and dendrograms
    """
    
    def __init__(self, country_filter: str = 'World'):
        """
        Initialize FactorTransformer with country filter.
        
        Args:
            country_filter: One of 'DM', 'EM', 'Asia', 'Europe', 'LatAm', 'World'
        """
        self.factor_map = {
            'PE': 'Valuation', 
            'Fwd_PE': 'Valuation', 
            'PB': 'Valuation',
            'PS': 'Valuation', 
            'Fwd_PS': 'Valuation', 
            'PCF': 'Valuation',
            'Fwd_PCF': 'Valuation', 
            'EV_EBIT': 'Valuation', 
            'EV_EBITDA': 'Valuation',
            'Fwd_EV_EBITDA': 'Valuation', 
            'EarningsYieldTTM':'Valuation',
            'EarningsYieldFWD': 'Valuation', 
            'CashFlowYieldTTM': 'Valuation',
            'CashFlowYieldFWD': 'Valuation', 
            'DVD': 'Valuation', 
            'Fwd_DVD': 'Valuation',

            # VALUATION SPREADS
            'EarningsYieldTTMSpread': 'Valuation', 
            'EarningsYieldFWDSpread': 'Valuation',
            'CashFlowYieldTTMSpread': 'Valuation', 
            'CashFlowYieldFWDSpread': 'Valuation',
            'DvdYieldTTMSpread': 'Valuation', 
            'DvdYieldFWDSpread': 'Valuation',

            # QUALITY FACTORS
            'Debt_to_Equity': 'Quality', 
            'Net_Debt_Ebitda': 'Quality',
            'AssetsEquity': 'Quality', 
            'Debt': 'Quality', 
            'Equity': 'Quality',
            'Liabilities': 'Quality', 
            'CF': 'Quality', 
            'FwdCF': 'Quality',
            'EbitMargin': 'Profitability', 
            'EbitdaMargin': 'Profitability',
            'NetMargin': 'Profitability', 
            'FwdEBITDAMargin': 'Profitability',
            'FwdNetMargin': 'Profitability', 
            'EBIT': 'Profitability',
            'EBITDA': 'Profitability', 
            'FwdEBITDA': 'Profitability',
            'EPS': 'Profitability', 
            'Earnings': 'Profitability',
            'FwdEarnings': 'Profitability', 
            'ROE': 'Profitability',
            'Fwd_ROE': 'Profitability', 
            'Return_Capital': 'Profitability',
            'ConsensusSalesGrowth': 'Profitability', 
            'ConsensusEbitdaGrowth': 'Profitability',
            'ConsensusEarningsGrowth': 'Profitability', 
            'ConsensusCashFlowGrowth': 'Profitability',
            'RollingEarnings': 'Momentum', 
            'FwdRollingEarnings': 'Momentum',
            'CumFlow': 'Momentum', 
            'Flows': 'Momentum',
            'Market_Cap': 'Momentum', 
            'EV': 'Momentum', 
            'Revenue': 'Momentum',
            'FwdRevenue': 'Quality', 
            'Price': 'Momentum', 
            'Assets': 'Quality',
            'RollingVol': 'Momentum', 
            'Ten_Year': 'Momentum',
            'SI': 'Momentum', 
            'SI_Ratio': 'Momentum',
            'GDP': 'Momentum', 
            'M2': 'Momentum'
        }
        
        self.factor_direction = {
             # VALUATION FACTORS (Lower multiples = better, so INVERT)
             'PE': -1,                    # Lower P/E is cheaper/better
             'Fwd_PE': -1,               # Lower forward P/E is better
             'PB': -1,                   # Lower P/B is cheaper
             'PS': -1,                   # Lower P/S is cheaper
             'Fwd_PS': -1,               # Lower forward P/S is better
             'PCF': -1,                  # Lower P/CF is cheaper
             'Fwd_PCF': -1,              # Lower forward P/CF is better
             'EV_EBIT': -1,              # Lower EV/EBIT is cheaper
             'EV_EBITDA': -1,            # Lower EV/EBITDA is cheaper
             'Fwd_EV_EBITDA': -1,        # Lower forward EV/EBITDA is better
             'DVD': -1,                  # Lower dividend discount is better (higher yield)
             'Fwd_DVD': -1,              # Lower forward dividend discount is better
             'EarningsYieldTTM': 1,      # Higher earnings yield is better
             'EarningsYieldFWD': 1,      # Higher forward earnings yield is better
             'CashFlowYieldTTM': 1,      # Higher cash flow yield is better
             'CashFlowYieldFWD': 1,      # Higher forward cash flow yield is better
             'EarningsYieldTTMSpread': 1,
             'EarningsYieldFWDSpread': 1,
             'CashFlowYieldTTMSpread': 1,
             'CashFlowYieldFWDSpread': 1,
             'DvdYieldTTMSpread': 1,
             'DvdYieldFWDSpread': 1,
             'ROE': 1,                   # Higher ROE is better
             'Fwd_ROE': 1,               # Higher forward ROE is better
             'Return_Capital': 1,        # Higher return on capital is better
             'Debt_to_Equity': -1,       # Lower D/E ratio is better (less leverage risk)
             'Net_Debt_Ebitda': -1,      # Lower net debt/EBITDA is better (less leverage)
             'AssetsEquity': -1,         # Lower assets/equity could mean less leverage
             'Assets': 1,                # More assets could be good (size/scale)
             'Debt': -1,                 # Less absolute debt is generally better
             'Equity': 1,                # More equity is generally better
             'Liabilities': -1,          # Fewer liabilities is better
             'CF': 1,                    # Higher cash flow is better
             'FwdCF': 1,                 # Higher forward cash flow is better
             'EbitMargin': 1,            # Higher EBIT margin is better
             'EbitdaMargin': 1,          # Higher EBITDA margin is better
             'NetMargin': 1,             # Higher net margin is better
             'FwdEBITDAMargin': 1,       # Higher forward EBITDA margin is better
             'FwdNetMargin': 1,          # Higher forward net margin is better
             'EBIT': 1,                  # Higher EBIT is better
             'EBITDA': 1,                # Higher EBITDA is better
             'FwdEBITDA': 1,             # Higher forward EBITDA is better
             'EPS': 1,                   # Higher EPS is better
             'Earnings': 1,              # Higher earnings is better
             'FwdEarnings': 1,           # Higher forward earnings is better
             'ConsensusSalesGrowth': 1,      # Higher sales growth is better
             'ConsensusEbitdaGrowth': 1,     # Higher EBITDA growth is better
             'ConsensusEarningsGrowth': 1,   # Higher earnings growth is better
             'ConsensusCashFlowGrowth': 1,   # Higher cash flow growth is better
             'RollingEarnings': 1,           # Higher rolling earnings growth is better
             'FwdRollingEarnings': 1,        # Higher forward rolling earnings is better
             'CumFlow': 1,                   # Positive cumulative flows are better
             'Flows': 1,                     # Positive flows are better
             'Market_Cap': 1,            # Larger market cap (more liquidity, stability)
             'EV': 1,                    # Larger enterprise value
             'Revenue': 1,               # Higher revenue (scale)
             'FwdRevenue': 1,            # Higher forward revenue
             'Price': 1,                 # Price itself is neutral, but momentum positive
             'RollingVol': -1,           # Lower volatility is better (less risky)
             'Ten_Year': 1,              # Higher bond yields could mean higher risk premiums
             'SI': -1,                   # Lower short interest is better
             'SI_Ratio': -1,             # Lower short interest ratio is better
             'GDP': 1,                   # Higher GDP growth is better
             'M2': 1,                    # Money supply growth can be positive for assets
        }
        
        # Country groupings
        self.DM = [
            "United States",
            "Japan",
            "United Kingdom",
            "Canada",
            "France",
            "Switzerland",
            "Germany",
            "Australia",
            "Netherlands",
            "Ireland",
            "Denmark",
            "Sweden",
            "Spain",
            "Hong Kong",
            "Italy",
            "Singapore",
            "Finland",
            "Belgium",
            "Norway"
        ]

        self.EM = [
            "India",
            "Taiwan",
            "South Korea",
            "Brazil",
            "Mexico",
            "South Africa",
            "Israel",
            "Indonesia",
            "Thailand",
            "Malaysia",
            "Poland",
            "Chile",
            "Peru",
            "Colombia"
        ]
        
        self.Asia = [
            "Japan",
            "China",
            "India",
            "Taiwan",
            "Australia",
            "South Korea",
            "Hong Kong",
            "Singapore",
            "Israel",
            "Indonesia",
            "Thailand",
            "Malaysia"
        ]
        
        self.Europe = [
            "United Kingdom",
            "France",
            "Switzerland",
            "Germany",
            "Netherlands",
            "Ireland",
            "Denmark",
            "Sweden",
            "Spain",
            "Italy",
            "Finland",
            "Belgium",
            "Norway",
            "Poland"
        ]
        
        self.LatAm = [
            "Brazil",
            "Mexico",
            "Chile",
            "Peru",
            "Colombia"
        ]
        
        self.World = [
            "United States",
            "Japan",
            "United Kingdom",
            "Canada",
            "France",
            "Switzerland",
            "China",
            "Germany",
            "India",
            "Taiwan",
            "Australia",
            "Netherlands",
            "South Korea",
            "Ireland",
            "Denmark",
            "Sweden",
            "Spain",
            "Hong Kong",
            "Italy",
            "Brazil",
            "Singapore",
            "Mexico",
            "South Africa",
            "Finland",
            "Belgium",
            "Israel",
            "Indonesia",
            "Thailand",
            "Norway",
            "Malaysia",
            "Poland",
            "Chile",
            "Peru",
            "Colombia"
        ]
        
        # Validate filter and set selected countries
        valid_filters = ['DM', 'EM', 'Asia', 'Europe', 'LatAm', 'World']
        if country_filter not in valid_filters:
            raise ValueError(f"Invalid country_filter. Choose from: {valid_filters}")
        
        self.country_filter = country_filter
        self.selected_countries = getattr(self, country_filter)
        self.window = 63

    def calculate_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates an expanding z-score and converts it to a percentile
        using the standard normal CDF. This method avoids lookahead bias.
        
        Args:
            df: DataFrame with dates as index and countries as columns
            
        Returns:
            DataFrame with percentile scores (0.0 to 1.0)
        """
        expanding_mean = df.expanding(min_periods=self.window).mean().shift(1)
        expanding_std = df.expanding(min_periods=self.window).std().shift(1)
        zscores = (df - expanding_mean) / expanding_std
        percentiles = stats.norm.cdf(zscores)
        return pd.DataFrame(percentiles, index=df.index, columns=df.columns).iloc[self.window:]
    
    def calculate_absolute_percentile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Historical percentile rank for each country.
        
        Args:
            df: DataFrame with dates as index and countries as columns
            
        Returns:
            DataFrame with percentile ranks
        """
        return df.expanding().rank(pct=True, method='min').iloc[self.window:]
    
    def calculate_relative_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional rank at each date.
        
        Args:
            df: DataFrame with dates as index and countries as columns
            
        Returns:
            DataFrame with relative rankings
        """
        return df.rank(axis=1, pct=True, method='average').iloc[self.window:]
    
    def calculate_delta_percentile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Percentile of 63-day percent changes.
        
        Args:
            df: DataFrame with dates as index and countries as columns
            
        Returns:
            DataFrame with delta percentile scores
        """
        pct_change = df.pct_change(periods=self.window)
        return pct_change.expanding().rank(pct=True, method='average').iloc[self.window:]
    
    def transform_all(self, 
                      factor_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Apply all transformations to factor dataframes.
        
        Args:
            factor_dfs: Dictionary mapping factor names to DataFrames
            
        Returns:
            Dict with structure: {factor_name: {metric_type: df}}
        """
        # Filter dataframes to selected countries
        filtered_factor_dfs = {}
        for factor_name, df in factor_dfs.items():
            available_countries = [c for c in self.selected_countries if c in df.columns]
            if len(available_countries) == 0:
                print(f"Warning: No selected countries found in {factor_name}. Skipping.")
                continue
            filtered_factor_dfs[factor_name] = df[available_countries]
        
        results = {}
        for factor_name, df in filtered_factor_dfs.items():
            direction = self.factor_direction.get(factor_name, 1)
            
            if direction == -1:
                # Invert percentiles (lower values = better)
                all_metrics = {
                    'zscore': 1.0 - self.calculate_zscore(df),
                    'absolute_pct': 1.0 - self.calculate_absolute_percentile(df),
                    'relative_rank': 1.0 - self.calculate_relative_ranking(df),
                    'delta_pct': 1.0 - self.calculate_delta_percentile(df)
                }
            elif direction == 1:
                # Don't invert (higher values = better)
                all_metrics = {
                    'zscore': self.calculate_zscore(df),
                    'absolute_pct': self.calculate_absolute_percentile(df),
                    'relative_rank': self.calculate_relative_ranking(df),
                    'delta_pct': self.calculate_delta_percentile(df)
                }
            else:
                print(f"Warning: Invalid direction '{direction}' for {factor_name}. Defaulting to 1.")
                all_metrics = {
                    'zscore': self.calculate_zscore(df),
                    'absolute_pct': self.calculate_absolute_percentile(df),
                    'relative_rank': self.calculate_relative_ranking(df),
                    'delta_pct': self.calculate_delta_percentile(df)
                }
            
            results[factor_name] = all_metrics
        
        self.factor_results = results
        return results

    def calculate_weighted_average(self,
                                   factor_metrics_dict: Dict[str, Dict[str, pd.DataFrame]],
                                   weights: Dict[str, float]) -> Dict[str, pd.DataFrame]:
        """
        Calculates a weighted-average score from a dictionary of metric DataFrames.
    
        Args:
            factor_metrics_dict: The nested dict: {factor_name: {metric_type: df}}
            weights: A dict of weights, e.g., {'zscore': 0.25, 'absolute_pct': 0.25, ...}
                     The weights must sum to 1.0 for a true average.
    
        Returns:
            A dictionary: {factor_name: final_score_df}
        """
        if not np.isclose(sum(weights.values()), 1.0):
            print(f"Warning: Weights do not sum to 1.0 (Sum={sum(weights.values())}).")
        
        final_scores = {}
        for factor_name, metrics_dict in factor_metrics_dict.items():
            try:
                all_metrics_df = pd.concat(metrics_dict, axis=1)
            except pd.errors.InvalidIndexError:
                print(f"Error: Could not concatenate data for {factor_name}. "
                      "This can happen if data is duplicated or not found.")
                print(f"Metrics available: {list(metrics_dict.keys())}")
                continue
        
            weighted_df = all_metrics_df.copy()
            for metric_name, weight in weights.items():
                if metric_name in weighted_df.columns.get_level_values(0):
                    weighted_df[metric_name] = weighted_df[metric_name] * weight
                else:
                    print(f"Warning: Metric '{metric_name}' in weights not found for {factor_name}.")

            final_factor_score = weighted_df.groupby(level=1, axis=1).sum()
            final_scores[factor_name] = final_factor_score
        
        self.weighted_average_scores = final_scores
        return final_scores
        
    def aggregate_by_category(self, 
                             weighted_scores: Dict[str, pd.DataFrame]
                             ) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Aggregates factor scores into category-level scores (Valuation, Quality, Momentum, etc.).
        
        Args:
            weighted_scores: Output from calculate_weighted_average. 
                            Dict with structure: {factor_name: final_score_df}
        
        Returns:
            Tuple of two dictionaries:
            1. {category_name: aggregated_score_df} - Category scores with countries as columns
            2. {country_name: category_scores_df} - Country scores with categories as columns
        """
        category_factors = {}
        for factor_name, score_df in weighted_scores.items():
            category = self.factor_map.get(factor_name)
            if category is None:
                print(f"Warning: Factor '{factor_name}' not found in factor_map. Skipping.")
                continue
            if category not in category_factors:
                category_factors[category] = []
            category_factors[category].append(score_df)
        
        # Calculate mean for each category
        category_scores = {}
        for category, dfs in category_factors.items():
            aligned_sum = dfs[0].copy() * 0
            for df in dfs:
                aligned_sum = aligned_sum.add(df, fill_value=0)
            category_scores[category] = aligned_sum / len(dfs)
        
        # Create country-centric view
        all_countries = set()
        for df in category_scores.values():
            all_countries.update(df.columns)
        all_countries = sorted(list(all_countries))
        
        print(f"Debug: Number of unique countries found: {len(all_countries)}")
        
        country_scores = {}
        for country in all_countries:
            country_data = {}
            for category, df in category_scores.items():
                if country in df.columns:
                    country_data[category] = df[country]
                else:
                    country_data[category] = pd.Series(index=df.index, dtype=float)
            country_scores[country] = pd.DataFrame(country_data)
        
        self.category_scores = category_scores
        self.country_scores = country_scores
        return category_scores, country_scores

    def calculate_composite_score(self, 
                                  category_weights: Dict[str, float]
                                  ) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Calculates composite scores and category contributions for each country.
        
        Args:
            category_weights: Dict of category weights, e.g., 
                             {'Quality': 0.25, 'Valuation': 0.25, 'Profitability': 0.1, 'Momentum': 0.4}
        
        Returns:
            Tuple of:
            1. DataFrame with composite scores (index: dates, columns: countries)
            2. Dict[country_name: contribution_df] showing each category's contribution to composite score
        """
        if not hasattr(self, 'country_scores'):
            raise AttributeError("country_scores not found. Run aggregate_by_category() first.")
        
        if not np.isclose(sum(category_weights.values()), 1.0):
            print(f"Warning: Category weights do not sum to 1.0 (Sum={sum(category_weights.values())}).")
        
        composite_scores = {}
        contribution_dict = {}
        
        for country, scores_df in self.country_scores.items():
            weighted_scores = scores_df * pd.Series(category_weights)
            composite_scores[country] = weighted_scores.sum(axis=1)
            total_weighted = weighted_scores.sum(axis=1)
            contributions = weighted_scores.div(total_weighted, axis=0)
            contribution_dict[country] = contributions
        
        composite_df = pd.DataFrame(composite_scores)
        self.composite_scores = composite_df
        self.category_contributions = contribution_dict
        return composite_df, contribution_dict

    def normalize_and_rebase_contributions(self) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Normalizes composite scores cross-sectionally (min-max normalization) and rebases 
        category contributions to sum to the normalized score for each country-date.
        
        Returns:
            Tuple of:
            1. DataFrame with normalized composite scores (index: dates, columns: countries)
            2. Dict[country_name: rebased_contributions_df] where contributions sum to 
               the normalized score (instead of 1.0) for each date
            3. Dict[category_name: rebased_contributions_df] with countries as columns
        """
        if not hasattr(self, 'composite_scores'):
            raise AttributeError("composite_scores not found. Run calculate_composite_score() first.")
        if not hasattr(self, 'category_contributions'):
            raise AttributeError("category_contributions not found. Run calculate_composite_score() first.")
        
        # Step 1: Normalize composite scores cross-sectionally for each date
        normalized_scores = self.composite_scores.copy()
        for date in normalized_scores.index:
            date_values = normalized_scores.loc[date]
            min_val = date_values.min()
            max_val = date_values.max()
            
            if max_val != min_val:
                normalized_scores.loc[date] = (date_values - min_val) / (max_val - min_val)
            else:
                print(f"WARNING: All composite scores are identical on {date}. "
                      f"Value: {min_val:.6f}. Assigning 0.5 to all countries.")
                normalized_scores.loc[date] = 0.5
        
        # Step 2: Rebase category contributions by country
        rebased_contributions_by_country = {}
        for country in self.category_contributions.keys():
            original_contributions = self.category_contributions[country].copy()
            rebased = original_contributions.copy()
            
            for date in rebased.index:
                if date in normalized_scores.index and country in normalized_scores.columns:
                    norm_score = normalized_scores.loc[date, country]
                    rebased.loc[date] = original_contributions.loc[date] * norm_score
            
            rebased_contributions_by_country[country] = rebased
        
        # Step 3: Pivot to category-centric view
        rebased_contributions_by_factor = {}
        sample_country = list(rebased_contributions_by_country.keys())[0]
        all_categories = rebased_contributions_by_country[sample_country].columns.tolist()
        
        for category in all_categories:
            category_data = {}
            for country, contrib_df in rebased_contributions_by_country.items():
                if category in contrib_df.columns:
                    category_data[country] = contrib_df[category]
            rebased_contributions_by_factor[category] = pd.DataFrame(category_data)
        
        self.normalized_scores = normalized_scores
        self.rebased_contributions_by_country = rebased_contributions_by_country
        self.rebased_contributions_by_factor = rebased_contributions_by_factor
        
        return normalized_scores, rebased_contributions_by_country, rebased_contributions_by_factor
    
    def calculate_factor_contribution_changes(self, 
                                             period: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Calculates the change in rebased factor contributions over a specified period.
        
        Args:
            period: Number of working days between measurements (default: 5)
        
        Returns:
            Dict[category_name: changes_df] with differences in contributions across countries
        """
        if not hasattr(self, 'rebased_contributions_by_factor'):
            raise AttributeError("rebased_contributions_by_factor not found. "
                               "Run normalize_and_rebase_contributions() first.")
        
        factor_changes = {}
        change_dates = None
        
        for category, contrib_df in self.rebased_contributions_by_factor.items():
            all_dates = contrib_df.index
            selected_dates = sorted(all_dates[::-period])
            changes = contrib_df.loc[selected_dates].diff()
            changes = changes.iloc[1:]
            factor_changes[category] = changes


            print(contrib_df.index[-1])
            if change_dates is None:
                change_dates = changes.index.tolist()

        # Calculate total factor changes by summing all categories
        dfs_list = list(factor_changes.values())
        total_changes = dfs_list[0].copy()
        for df in dfs_list[1:]:
            total_changes = total_changes.add(df, fill_value=0)

        self.factor_contribution_changes = factor_changes
        self.total_factor_changes = total_changes        
        self.change_dates = change_dates
        

        
        print(f"Calculated {len(change_dates)} change periods with {period}-day intervals.")
        print(f"First change date: {change_dates[0]}, Last change date: {change_dates[-1]}")
        
        return factor_changes, total_changes
            
    def plot_factor_contributions(self, 
                                  date: str = None,
                                  figsize: tuple = (12, 6)) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Plots factor contributions and changes as stacked bar charts for a specific date.
        
        Args:
            date: Date to plot in string format (e.g., '2024-01-15'). 
                  If None, uses the last available date.
            figsize: Figure size (default: (12, 6))
        
        Returns:
            Tuple of:
            1. DataFrame with normalized scores (index: categories, columns: countries)
            2. DataFrame with contribution changes (index: categories, columns: countries)
        """
        if not hasattr(self, 'rebased_contributions_by_factor'):
            raise AttributeError("rebased_contributions_by_factor not found. "
                               "Run normalize_and_rebase_contributions() first.")
        if not hasattr(self, 'factor_contribution_changes'):
            raise AttributeError("factor_contribution_changes not found. "
                               "Run calculate_factor_contribution_changes() first.")
        
        sample_category = list(self.rebased_contributions_by_factor.keys())[0]
        available_dates = self.rebased_contributions_by_factor[sample_category].index
        
        if date is None:
            plot_date = available_dates[-1]
        else:
            plot_date = pd.to_datetime(date)
            if plot_date not in available_dates:
                raise ValueError(f"Date {date} not found in rebased contributions. "
                               f"Available range: {available_dates[0]} to {available_dates[-1]}")
        
        if plot_date not in self.change_dates:
            raise ValueError(f"Date {plot_date.strftime('%Y-%m-%d')} not found in change_dates. "
                           f"Available change dates: {len(self.change_dates)} dates from "
                           f"{self.change_dates[0].strftime('%Y-%m-%d')} to {self.change_dates[-1].strftime('%Y-%m-%d')}")
        
        contributions_data = {}
        changes_data = {}
        for category in self.rebased_contributions_by_factor.keys():
            contributions_data[category] = self.rebased_contributions_by_factor[category].loc[plot_date]
            changes_data[category] = self.factor_contribution_changes[category].loc[plot_date]
        
        contributions_df = pd.DataFrame(contributions_data).T
        changes_df = pd.DataFrame(changes_data).T
        
        total_scores = contributions_df.sum(axis=0)
        sorted_countries = total_scores.sort_values(ascending=False).index
        contributions_df = contributions_df[sorted_countries]
        changes_df = changes_df[sorted_countries]
        
        category_colors = {
            list(contributions_df.index)[0]: '#1F3864',
            list(contributions_df.index)[1]: '#38E2E6',
            list(contributions_df.index)[2]: '#CCBD66',
            list(contributions_df.index)[3]: '#0099FF'
        }
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                         gridspec_kw={'height_ratios': [2, 1]},
                                         constrained_layout=True)
        
        countries = contributions_df.columns
        x_pos = np.arange(len(countries))
        bottom1 = np.zeros(len(countries))
        
        for category in contributions_df.index:
            values = contributions_df.loc[category].values
            ax1.bar(x_pos, values, bottom=bottom1, 
                   label=category, color=category_colors[category])
            bottom1 += values
        
        ax1.set_ylabel('Normalized Score', fontsize=10)
        ax1.set_title(f'Normalized Equity Ranking - {plot_date.strftime("%Y-%m-%d")}', 
                     fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_xlim(-0.5, len(countries) - 0.5)
        
        bottom2 = np.zeros(len(countries))
        for category in changes_df.index:
            values = changes_df.loc[category].values
            ax2.bar(x_pos, values, bottom=bottom2, 
                   label=category, color=category_colors[category])
            bottom2 += values
        
        ax2.set_ylabel('Contribution Changes', fontsize=10)
        ax2.set_xlabel('Countries', fontsize=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(countries, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_xlim(-0.5, len(countries) - 0.5)
        
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1),
                  frameon=False, fontsize=10)
        
        plt.show()
        
        self.plot_contributions_df = contributions_df
        self.plot_changes_df = changes_df
        
        return contributions_df, changes_df            
    
    def analyze_factor_redundancy(self, 
                                  distance_threshold: float = 0.3,
                                  linkage_method: str = 'ward',
                                  selection_criterion: str = 'coverage') -> Dict:
        """
        Analyzes redundancy between factors using hierarchical clustering on correlations.
        
        Args:
            distance_threshold: Threshold for cutting dendrogram (0-1). Lower = fewer, more distinct factors.
            linkage_method: Method for hierarchical clustering ('ward', 'average', 'complete', 'single')
            selection_criterion: How to select representative from each cluster:
                               'coverage' - Factor with most data availability
                               'unique' - Factor with lowest correlation to other clusters
                               'central' - Factor most correlated with its cluster members
        
        Returns:
            Dict containing:
            - 'correlation_matrix': DataFrame of factor correlations
            - 'distance_matrix': Distance matrix used for clustering
            - 'linkage': Linkage matrix for dendrogram
            - 'clusters': Dict[cluster_id: list of factors]
            - 'recommended_factors': List of selected factors
            - 'selection_rationale': Dict[factor: reason for selection]
        """
        if not hasattr(self, 'weighted_average_scores'):
            raise AttributeError("weighted_average_scores not found. "
                               "Run calculate_weighted_average() first.")
        
        self.clustering_threshold = distance_threshold
        
        factor_names = list(self.weighted_average_scores.keys())
        n_factors = len(factor_names)
        
        correlation_matrix = pd.DataFrame(
            np.zeros((n_factors, n_factors)),
            index=factor_names,
            columns=factor_names
        )
        
        print(f"Calculating correlations for {n_factors} factors...")
        
        for i, factor1 in enumerate(factor_names):
            for j, factor2 in enumerate(factor_names):
                if i <= j:
                    df1 = self.weighted_average_scores[factor1]
                    df2 = self.weighted_average_scores[factor2]
                    
                    country_correlations = []
                    for country in df1.columns:
                        if country in df2.columns:
                            combined = pd.DataFrame({
                                'f1': df1[country],
                                'f2': df2[country]
                            }).dropna()
                            
                            if len(combined) > 10:
                                corr = combined['f1'].corr(combined['f2'])
                                if not np.isnan(corr):
                                    country_correlations.append(corr)
                    
                    avg_corr = np.mean(country_correlations) if country_correlations else 0
                    correlation_matrix.loc[factor1, factor2] = avg_corr
                    correlation_matrix.loc[factor2, factor1] = avg_corr
        
        distance_matrix = 1 - np.abs(correlation_matrix.values)
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method=linkage_method)
        cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
        
        clusters = {}
        for factor, cluster_id in zip(factor_names, cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(factor)
        
        print(f"\nIdentified {len(clusters)} factor clusters:")
        for cluster_id, factors in clusters.items():
            print(f"  Cluster {cluster_id}: {len(factors)} factors - {factors[:3]}{'...' if len(factors) > 3 else ''}")
        
        recommended_factors = []
        selection_rationale = {}
        
        for cluster_id, factors_in_cluster in clusters.items():
            if len(factors_in_cluster) == 1:
                selected = factors_in_cluster[0]
                rationale = "Only factor in cluster"
            elif selection_criterion == 'coverage':
                coverage_scores = {}
                for factor in factors_in_cluster:
                    df = self.weighted_average_scores[factor]
                    coverage = df.notna().sum().sum() / (len(df) * len(df.columns))
                    coverage_scores[factor] = coverage
                selected = max(coverage_scores, key=coverage_scores.get)
                rationale = f"Best data coverage: {coverage_scores[selected]:.2%}"
            elif selection_criterion == 'unique':
                uniqueness_scores = {}
                other_clusters_factors = [f for cid, flist in clusters.items() 
                                         if cid != cluster_id for f in flist]
                
                for factor in factors_in_cluster:
                    if other_clusters_factors:
                        correlations = [abs(correlation_matrix.loc[factor, other_f]) 
                                      for other_f in other_clusters_factors]
                        uniqueness_scores[factor] = np.mean(correlations)
                    else:
                        uniqueness_scores[factor] = 0
                
                selected = min(uniqueness_scores, key=uniqueness_scores.get)
                rationale = f"Most unique (avg corr to other clusters: {uniqueness_scores[selected]:.3f})"
            elif selection_criterion == 'central':
                centrality_scores = {}
                for factor in factors_in_cluster:
                    correlations = [abs(correlation_matrix.loc[factor, other_f]) 
                                  for other_f in factors_in_cluster if other_f != factor]
                    centrality_scores[factor] = np.mean(correlations) if correlations else 0
                
                selected = max(centrality_scores, key=centrality_scores.get)
                rationale = f"Most central (avg corr within cluster: {centrality_scores[selected]:.3f})"
            else:
                raise ValueError(f"Invalid selection_criterion: {selection_criterion}")
            
            recommended_factors.append(selected)
            selection_rationale[selected] = rationale
        
        print(f"\nRecommended {len(recommended_factors)} factors:")
        for factor in recommended_factors:
            print(f"  {factor}: {selection_rationale[factor]}")
        
        results = {
            'correlation_matrix': correlation_matrix,
            'distance_matrix': pd.DataFrame(distance_matrix, 
                                            index=factor_names, 
                                            columns=factor_names),
            'linkage': linkage_matrix,
            'clusters': clusters,
            'recommended_factors': recommended_factors,
            'selection_rationale': selection_rationale,
            'factor_names': factor_names
        }
        
        self.redundancy_analysis = results
        return results
    
    def plot_factor_dendrogram(self, 
                               figsize: tuple = (14, 8),
                               threshold_line: bool = True) -> None:
        """
        Plots dendrogram from factor redundancy analysis.
        
        Args:
            figsize: Figure size
            threshold_line: Whether to show the distance threshold line
        """
        if not hasattr(self, 'redundancy_analysis'):
            raise AttributeError("redundancy_analysis not found. "
                               "Run analyze_factor_redundancy() first.")
        if not hasattr(self, 'clustering_threshold'):
            raise AttributeError("clustering_threshold not found. "
                               "Run analyze_factor_redundancy() first.")
        
        results = self.redundancy_analysis
        
        fig, ax = plt.subplots(figsize=figsize)
        
        dendro = dendrogram(
            results['linkage'],
            labels=results['factor_names'],
            ax=ax,
            leaf_rotation=90,
            leaf_font_size=9
        )
        
        ax.set_title('Factor Redundancy Dendrogram', fontsize=14, fontweight='bold')
        ax.set_xlabel('Factors', fontsize=11)
        ax.set_ylabel('Distance (1 - |correlation|)', fontsize=11)
        
        if threshold_line:
            ax.axhline(y=self.clustering_threshold, color='r', linestyle='--', 
                      linewidth=2, label=f'Threshold = {self.clustering_threshold}')
            ax.legend()
        
        plt.tight_layout()
        plt.show()

