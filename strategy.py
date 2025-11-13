"""
Country Rotation Strategy - Main Script

This script implements a quantitative country rotation strategy using financial and economic data.
It processes multiple datasets, performs regional aggregations, calculates derived metrics,
and exports the results for further analysis.

All functions are imported from function_module.py for better code organization.
"""

import os
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List, Tuple, Any, Optional
warnings.filterwarnings('ignore')

# Set working directory
os.chdir('D:/Users/avazquez/OneDrive - valmexcasabolsa/Documents/QuantModels/country_rotation')

# Import all functions from the function module

import function_module as fm

warnings.filterwarnings('ignore')
#%%
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
        # ==========================================
        # STEP 1: LOAD RAW DATA
        # ==========================================
        print("\n🔄 STEP 1: Loading raw data from Excel files...")
        
        dataFrames = fm.read_excel_files_to_dict('Inputs')
        fileNames = list(dataFrames.keys())
        
        print(f"✅ Successfully loaded {len(dataFrames)} datasets:")
        for i, name in enumerate(fileNames[:5], 1):  # Show first 5
            print(f"   {i}. {name}")
        if len(fileNames) > 5:
            print(f"   ... and {len(fileNames) - 5} more datasets")
        
        # Quick data check
        if 'Assets' in dataFrames:
            print(f"📊 Sample data shape (Assets): {dataFrames['Assets'].shape}")
        
        # ==========================================
        # STEP 2: LOAD CLASSIFICATION DATA
        # ==========================================
        print("\n🔄 STEP 2: Loading classification data...")
        
        classification, classification_metricas, classification_map = fm.load_classification_data()
        
        print(f"✅ Classification data loaded:")
        print(f"   📍 Countries: {len(classification)}")
        print(f"   📊 Metrics mapping: {len(classification_map)} entries")
        
        # ==========================================
        # STEP 3: DATA VALIDATION
        # ==========================================
        print("\n🔄 STEP 3: Validating input data...")
        
        fm.validate_inputs(dataFrames, classification)
        
        # ==========================================
        # STEP 4: REMOVE WEEKENDS
        # ==========================================
        print("\n🔄 STEP 4: Removing weekends from time series...")
        
        dataFrames = fm.remove_weekends_optimized(dataFrames)
        
        print("✅ Weekend removal completed for all datasets")
        
        # ==========================================
        # STEP 5: SLICE DATA AND DROP COUNTRIES
        # ==========================================
        print("\n🔄 STEP 5: Slicing data and filtering countries...")
        
        # Default: slice from 2010-01-01 and drop Saudi Arabia
        dataFrames = fm.slice_data_frames_by_date(
            dataFrames, 
            target_date='2010-01-01',
            columns_to_drop=['Saudi Arabia']
        )
        
        print("✅ Data slicing and country filtering completed")
        
        # ==========================================
        # STEP 6: CREATE REGIONAL CLASSIFICATIONS
        # ==========================================
        print("\n🔄 STEP 6: Creating regional classifications...")
        
        regions_dict = fm.get_regions_dict(classification)
        
        print(f"✅ Regional classifications created:")
        for region, countries in regions_dict.items():
            print(f"   🌍 {region}: {len(countries)} countries")
        
        # ==========================================
        # STEP 7: COMPREHENSIVE DATA PROCESSING
        # ==========================================
        print("\n🔄 STEP 7: Processing and transforming financial data...")
        print("This includes:")
        print("   • Regional aggregations")
        print("   • Growth metrics calculation")
        print("   • Yield computations")
        print("   • Spread analysis")
        print("   • Rolling statistics")
        print("   • Balance sheet ratios")
        
        processed_data = fm.transform_process_data(
            dataFrames, 
            classification, 
            output_folder='ProcessedInputs'
        )
        
        # ==========================================
        # STEP 8: FINAL SUMMARY
        # ==========================================
        print("\n" + "=" * 60)
        print("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\n📈 PROCESSING SUMMARY:")
        print(f"   • Original datasets: {len(fileNames)}")
        print(f"   • Final processed datasets: {len(processed_data)}")
        print(f"   • Countries in classification: {len(classification)}")
        print(f"   • Regional groupings: {len(regions_dict)}")
        print(f"   • Export folder: ProcessedInputs/")
        
        # Show some key metrics that were created
        derived_metrics = [k for k in processed_data.keys() if k not in fileNames]
        if derived_metrics:
            print(f"\n🔧 NEW DERIVED METRICS ({len(derived_metrics)}):")
            for i, metric in enumerate(derived_metrics[:10], 1):  # Show first 10
                print(f"   {i}. {metric}")
            if len(derived_metrics) > 10:
                print(f"   ... and {len(derived_metrics) - 10} more derived metrics")
        
        print(f"\n✅ All processed data exported to: ProcessedInputs/")
        print(f"✅ Ready for quantitative analysis and strategy implementation!")
        
        return processed_data, regions_dict, classification
        
    except Exception as e:
        print(f"\n❌ ERROR IN PIPELINE EXECUTION:")
        print(f"   {str(e)}")
        print(f"\n🔧 Please check your data files and try again.")
        raise

# Execute the main pipeline
processed_data, regions_dict, classification = run_inputs()

# Make key variables available in the global scope for interactive use
dataFrames = processed_data

print(f"\n📋 VARIABLES AVAILABLE FOR ANALYSIS:")
print(f"   • dataFrames: {len(dataFrames)} processed datasets")
print(f"   • regions_dict: Regional country groupings")
print(f"   • classification: Country classification DataFrame")
print(f"   • processed_data: Same as dataFrames (alias)")

#Quick data exploration
fm.explore_data(dataFrames, regions_dict)


 
 #%%

class FactorTransformer:
    """
    Transforms factor data into standardized metrics.
    """
    
    def __init__(self, country_filter: str = 'World'):
        
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
        
        self.factor_direction ={
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
        
        
        # Add filter validation
        valid_filters = ['DM', 'EM', 'Asia', 'Europe', 'LatAm', 'World']
        if country_filter not in valid_filters:
            raise ValueError(f"Invalid country_filter. Choose from: {valid_filters}")
        
        self.country_filter = country_filter
        self.selected_countries = getattr(self, country_filter)
        self.window = 63

    def calculate_zscore(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates an expanding z-score and converts it to a percentile
        using the standard normal CDF. This method avoids lookahead bias.
        """
        # min_periods=min_window will return NaN until the window is full
        # The shift(1) ensures data at time `t` is compared to history up to `t-1`
        expanding_mean = df.expanding(min_periods=self.window).mean().shift(1)
        expanding_std = df.expanding(min_periods=self.window).std().shift(1)
        # This is the original z-score calculation
        zscores = (df - expanding_mean) / expanding_std    
        # Convert z-scores to percentiles (0.0 to 1.0)
        # norm.cdf() works element-wise on the DataFrame
        percentiles = stats.norm.cdf(zscores)#.iloc[self.window]
        return pd.DataFrame(percentiles, index=df.index, columns= df.columns).iloc[self.window:]
    
    def calculate_absolute_percentile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Historical percentile rank for each country."""
        return df.expanding().rank(pct=True, method='min').iloc[self.window:]
    
    def calculate_relative_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank at each date."""
        return df.rank(axis=1, pct=True, method='average').iloc[self.window:]
    
    def calculate_delta_percentile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Percentile of 63-day percent changes."""
        pct_change = df.pct_change(periods=self.window)
        return pct_change.expanding().rank(pct=True, method='average').iloc[self.window:]
    
    
    def transform_all(self, 
                      factor_dfs: Dict[str, pd.DataFrame], 
                      ) -> Dict[str, Dict[str, pd.DataFrame]]:
        
        ## Here we can filter countries.
        
        
        """
        Apply all transformations to factor dataframes.
        
        Returns:
            Dict with structure: {factor_name: {metric_type: df}}
        """
        
        # At the start, filter each dataframe to selected countries
        filtered_factor_dfs = {}
    
        for factor_name, df in factor_dfs.items():
            # Get intersection of available columns and selected countries
            available_countries = [c for c in self.selected_countries if c in df.columns]
            
            if len(available_countries) == 0:
                print(f"Warning: No selected countries found in {factor_name}. Skipping.")
                continue
                
            filtered_factor_dfs[factor_name] = df[available_countries]
        
        results = {}
        
        for factor_name, df in filtered_factor_dfs.items():
            
            direction = self.factor_direction.get(factor_name, 1) # Get the direction for this factor. Default to 1 (high=good) if not specified.
                
            if direction == -1:
                # This inverts the percentiles.
                #High is bad
                # 95th percentile (bad) becomes 5th percentile (good).
                # 10th percentile (good) becomes 90th percentile (bad).
                
                all_metrics={
                    'zscore': 1.0 - self.calculate_zscore(df),
                    'absolute_pct': 1.0 - self.calculate_absolute_percentile(df),
                    'relative_rank': 1.0 - self.calculate_relative_ranking(df),
                    'delta_pct': 1.0 - self.calculate_delta_percentile(df)
                }
            
            elif direction == 1:
            # This does not inverts the percentiles.
            # High is good
            # 95th percentile (good) becomes 5th percentile (bad).
            # 10th percentile (bad) becomes 90th percentile (good).
                all_metrics={
                    'zscore': self.calculate_zscore(df),
                    'absolute_pct': self.calculate_absolute_percentile(df),
                    'relative_rank': self.calculate_relative_ranking(df),
                    'delta_pct': self.calculate_delta_percentile(df)
                }
            
            elif direction != 1:
                print(f"Warning: Invalid direction '{direction}' for {factor_name}. Defaulting to 1.")
                # No change needed if direction is 1
            
            results[factor_name] = all_metrics # Store metrics
        
        self.factor_results = results

        return results

    def calculate_weighted_average(self,
        factor_metrics_dict: Dict[str, Dict[str, pd.DataFrame]],
        weights: Dict[str, float]
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates a weighted-average score from a dictionary of metric DataFrames.
    
        Args:
            factor_metrics_dict: The nested dict: {factor_name: {metric_type: df}}
            weights: A dict of weights, e.g., {'zscore': 0.25, 'absolute_pct': 0.25, ...}
                     The weights must sum to 1.0 for a true average.
    
        Returns:
            A dictionary: {factor_name: final_score_df}
        """
        
        
        # Optional: Check if weights sum to 1.0
        if not np.isclose(sum(weights.values()), 1.0):
            print(f"Warning: Weights do not sum to 1.0 (Sum={sum(weights.values())}).")
            
        final_scores = {}
        
        for factor_name, metrics_dict in factor_metrics_dict.items():
            # metrics_dict looks like: {'zscore': df, 'absolute_pct': df, ...}
    
            # --- Step 1: Align all 4 metrics into one DataFrame ---
            # This aligns all DataFrames by their index (date).
            # It creates a MultiIndex for the columns: (metric_type, country)
            # e.g., ('zscore', 'USA'), ('zscore', 'CAN'), ('absolute_pct', 'USA'), ...
            try:
                all_metrics_df = pd.concat(metrics_dict, axis=1)
            except pd.errors.InvalidIndexError as e:
                print(f"Error: Could not concatenate data for {factor_name}. "
                      "This can happen if data is duplicated or not found.")
                print(f"Metrics available: {list(metrics_dict.keys())}")
                continue # Skip this factor
        
            # --- Step 3: Apply weights ---
            # Create a copy to avoid SettingWithCopyWarning
            weighted_df = all_metrics_df.copy()
            
            for metric_name, weight in weights.items():
                if metric_name in weighted_df.columns.get_level_values(0):
                    # This operation multiplies all columns under the metric_name
                    # (e.g., 'zscore') by its weight.
                    weighted_df[metric_name] = weighted_df[metric_name] * weight
                else:
                    print(f"Warning: Metric '{metric_name}' in weights not found for {factor_name}.")
    
            # --- Step 4: Sum the weighted metrics for each country ---
            # We group by level=1 (the country level, e.g., 'USA', 'CAN')
            # and sum the weighted metrics (e.g., weighted_zscore + weighted_abs_pct + ...)
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
                            Each DataFrame has dates as index, countries as columns
        
        Returns:
            Tuple of two dictionaries:
            1. {category_name: aggregated_score_df} - Category scores with countries as columns
            2. {country_name: category_scores_df} - Country scores with categories as columns
        """
        
        category_factors = {}
        
        # Group factors by their category
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
            # Average across all factors in this category
            # Align by index (dates) and columns (countries), then take mean
            aligned_sum = dfs[0].copy() * 0  # Initialize with zeros, preserving structure
            
            for df in dfs:
                aligned_sum = aligned_sum.add(df, fill_value=0)
            
            category_scores[category] = aligned_sum / len(dfs)
        
        # Create country-centric view
        all_countries = set()
        for df in category_scores.values():
            all_countries.update(df.columns)
        
        all_countries = sorted(list(all_countries))  # Sort for consistency
        
        print(f"Debug: Number of unique countries found: {len(all_countries)}")
        
        country_scores = {}
        
        for country in all_countries:
            country_data = {}
            for category, df in category_scores.items():
                if country in df.columns:
                    country_data[category] = df[country]
                else:
                    # Fill with NaN if country not in this category
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
        
        # Optional: Check if weights sum to 1.0
        if not np.isclose(sum(category_weights.values()), 1.0):
            print(f"Warning: Category weights do not sum to 1.0 (Sum={sum(category_weights.values())}).")
        
        composite_scores = {}
        contribution_dict = {}
        
        for country, scores_df in self.country_scores.items():
            # Apply weights to each category
            weighted_scores = scores_df * pd.Series(category_weights)
            
            # Calculate composite score (sum of weighted categories)
            composite_scores[country] = weighted_scores.sum(axis=1)
            
            # Calculate contributions (weighted score / total weighted score)
            total_weighted = weighted_scores.sum(axis=1)
            contributions = weighted_scores.div(total_weighted, axis=0)
            
            contribution_dict[country] = contributions
        
        # Convert composite scores dict to DataFrame
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
               Each date is normalized cross-sectionally: (value - min) / (max - min)
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
            
            # Min-max normalization
            if max_val != min_val:  # Avoid division by zero
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
            
            # For each date, multiply contributions by the normalized score
            for date in rebased.index:
                if date in normalized_scores.index and country in normalized_scores.columns:
                    norm_score = normalized_scores.loc[date, country]
                    rebased.loc[date] = original_contributions.loc[date] * norm_score
            
            rebased_contributions_by_country[country] = rebased
        
        # Step 3: Pivot to category-centric view
        rebased_contributions_by_factor = {}
        
        # Get all unique categories from any country's contribution dataframe
        sample_country = list(rebased_contributions_by_country.keys())[0]
        all_categories = rebased_contributions_by_country[sample_country].columns.tolist()
        
        for category in all_categories:
            category_data = {}
            for country, contrib_df in rebased_contributions_by_country.items():
                if category in contrib_df.columns:
                    category_data[country] = contrib_df[category]
            
            rebased_contributions_by_factor[category] = pd.DataFrame(category_data)
        
        # Store results
        self.normalized_scores = normalized_scores
        self.rebased_contributions_by_country = rebased_contributions_by_country
        self.rebased_contributions_by_factor = rebased_contributions_by_factor
        
        return normalized_scores, rebased_contributions_by_country, rebased_contributions_by_factor
    
    def calculate_factor_contribution_changes(self, 
                                             period: int = 5
                                             ) -> Dict[str, pd.DataFrame]:
        """
        Calculates the change in rebased factor contributions over a specified period.
        
        Args:
            period: Number of working days between measurements (default: 5)
                    Changes are calculated strictly every N days, not rolling.
        
        Returns:
            Dict[category_name: changes_df] with differences in contributions across countries
        """
        
        if not hasattr(self, 'rebased_contributions_by_factor'):
            raise AttributeError("rebased_contributions_by_factor not found. "
                               "Run normalize_and_rebase_contributions() first.")
        
        factor_changes = {}
        change_dates = None
        
        for category, contrib_df in self.rebased_contributions_by_factor.items():
            # Get all dates
            all_dates = contrib_df.index
            
            # Select dates at every 'period' interval
            selected_dates = all_dates[::period]
            
            # Calculate differences between consecutive selected dates
            changes = contrib_df.loc[selected_dates].diff()
            
            # Remove first row (NaN from diff)
            changes = changes.iloc[1:]
            
            factor_changes[category] = changes
            
            # Store dates (same for all categories)
            if change_dates is None:
                change_dates = changes.index.tolist()
        
        # Store results
        self.factor_contribution_changes = factor_changes
        self.change_dates = change_dates
        
        print(f"Calculated {len(change_dates)} change periods with {period}-day intervals.")
        print(f"First change date: {change_dates[0]}, Last change date: {change_dates[-1]}")
        
        return factor_changes
    
            
    def plot_factor_contributions(self, 
                                  date: str = None,
                                  figsize: tuple = (12, 6)
                                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        
        # Get the date to plot
        sample_category = list(self.rebased_contributions_by_factor.keys())[0]
        available_dates = self.rebased_contributions_by_factor[sample_category].index
        
        if date is None:
            plot_date = available_dates[-1]
        else:
            plot_date = pd.to_datetime(date)
            if plot_date not in available_dates:
                raise ValueError(f"Date {date} not found in rebased contributions. "
                               f"Available range: {available_dates[0]} to {available_dates[-1]}")
        
        # Check if date exists in change_dates
        if plot_date not in self.change_dates:
            raise ValueError(f"Date {plot_date.strftime('%Y-%m-%d')} not found in change_dates. "
                           f"Available change dates: {len(self.change_dates)} dates from "
                           f"{self.change_dates[0].strftime('%Y-%m-%d')} to {self.change_dates[-1].strftime('%Y-%m-%d')}")
        
        # Extract data for the specific date
        contributions_data = {}
        changes_data = {}
        
        for category in self.rebased_contributions_by_factor.keys():
            contributions_data[category] = self.rebased_contributions_by_factor[category].loc[plot_date]
            changes_data[category] = self.factor_contribution_changes[category].loc[plot_date]
        
        # Create DataFrames (index: categories, columns: countries)
        contributions_df = pd.DataFrame(contributions_data).T
        changes_df = pd.DataFrame(changes_data).T
        
        # Sort countries by total normalized score (descending)
        total_scores = contributions_df.sum(axis=0)
        sorted_countries = total_scores.sort_values(ascending=False).index
        
        contributions_df = contributions_df[sorted_countries]
        changes_df = changes_df[sorted_countries]
        
        # Define colors for categories
        category_colors = {
            list(contributions_df.index)[0]: '#1F3864',
            list(contributions_df.index)[1]: '#38E2E6',
            list(contributions_df.index)[2]: '#CCBD66',
            list(contributions_df.index)[3]: '#0099FF'
        }
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                         gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Stacked bar chart of contributions
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
        
        # Plot 2: Stacked bar chart of changes
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
        
        # Add legend outside plot area
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), 
                  frameon=False, fontsize=10)
        
        plt.tight_layout()
        #plt.savefig('equity_ranking.png', bbox_inches='tight', dpi=300)
        plt.show()
        
        # Store DataFrames
        self.plot_contributions_df = contributions_df
        self.plot_changes_df = changes_df
        
        return contributions_df, changes_df            
                
#%%

weights = {
'zscore': 0.25,
'absolute_pct': 0.25,
'relative_rank': 0.1,
'delta_pct': 0.4
}
        
category_weights = {
'Quality': 0.25,
'Valuation': 0.1,
'Profitability': 0.35,
'Momentum': 0.3
}

factor= FactorTransformer('DM')
#%%
results= factor.transform_all(dataFrames)
final_scores= factor.calculate_weighted_average(results, weights)
category_scores, country_scores = factor.aggregate_by_category(final_scores)
composite_scores, category_contributions = factor.calculate_composite_score(category_weights)
normalized_scores, rebased_contributions_by_country, rebased_contributions_by_factor = factor.normalize_and_rebase_contributions()
factor_changes = factor.calculate_factor_contribution_changes()
#%%

for country in normalized_scores.columns:
    normalized_scores[country].plot(title=country.upper(), figsize=(12,6))
    plt.show()


#%%
plt.style.use('seaborn-v0_8')
factor.plot_factor_contributions()
