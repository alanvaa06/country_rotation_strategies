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
    
    def __init__(self):
        
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
            'Ten_Year': 'Macro',
            'SI': 'Momentum', 
            'SI_Ratio': 'Momentum',
            'GDP': 'Macro', 
            'M2': 'Macro'
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
        """
        Apply all transformations to factor dataframes.
        
        Returns:
            Dict with structure: {factor_name: {metric_type: df}}
        """
        results = {}
        
        for factor_name, df in factor_dfs.items():
            
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
    
        return final_scores



#%%

weights = {
'zscore': 0.25,
'absolute_pct': 0.25,
'relative_rank': 0.1,
'delta_pct': 0.4
}
        
#%%        
factor= FactorTransformer()
#%%
results= factor.transform_all(dataFrames)
#%%
final_scores= factor.calculate_weighted_average(results, weights)

#%%
for 
final_scores


















