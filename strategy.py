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



# Import all functions from the function module
import function_module as fm

warnings.filterwarnings('ignore')

#%%
# Set working directory
os.chdir('D:/Users/avazquez/OneDrive - valmexcasabolsa/Documents/QuantModels/country_rotation')

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
# Create and configure the transformer
transformer = fm.QuantMetricsTransformer()
    
# Example usage (assuming you have your dataframes ready)
transformer.load_data(dataFrames).transform().validate().generate_report()

# Get results
transformed_dataFrames, new_classification = transformer.get_results()


#%%

class CountryFactorSelectionFramework:
    """
    Advanced framework for factor selection and correlation filtering for country-level analysis.
    
    This class handles DataFrames dictionary where each key is a metric name and each DataFrame
    contains time series data across different countries. The analysis standardizes metrics
    across countries and performs factor selection at the country cross-section level.
    
    NEW: Includes signal directionality correction to ensure all factors point in the same direction.
    """
    
    def __init__(self, correlation_threshold: float = 0.85, 
                 vif_threshold: float = 5.0,
                 min_data_coverage: float = 0.6):
        """
        Initialize the country-level factor selection framework.
        
        Parameters:
        -----------
        correlation_threshold : float, default 0.85
            Maximum allowed correlation between factors
        vif_threshold : float, default 5.0
            Maximum allowed Variance Inflation Factor
        min_data_coverage : float, default 0.6
            Minimum data coverage required per metric (60%)
        """
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.min_data_coverage = min_data_coverage
        self.results = {}
        
    def create_classification_map(self) -> Dict[str, str]:
        """
        Create a comprehensive factor classification map based on financial theory.
        
        Returns:
        --------
        Dict[str, str] : Factor to category mapping
        """
        
        factor_category = {
            # VALUATION FACTORS
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
            'EarningsYieldTTM': 'Valuation',
            'EarningsYieldFWD': 'Valuation',
            'CashFlowYieldTTM': 'Valuation',
            'CashFlowYieldFWD': 'Valuation',
            'DVD': 'Valuation',
            'Fwd_DVD': 'Valuation',
            
            # VALUATION SPREADS (Risk Premium)
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
            
            # PROFITABILITY FACTORS
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
            
            # MOMENTUM FACTORS  
            'ConsensusSalesGrowth': 'Momentum',
            'ConsensusEbitdaGrowth': 'Momentum',
            'ConsensusEarningsGrowth': 'Momentum', 
            'ConsensusCashFlowGrowth': 'Momentum',
            'RollingEarnings': 'Momentum',
            'FwdRollingEarnings': 'Momentum',
            'CumFlow': 'Momentum',
            'Flows': 'Momentum',
            
            # SIZE FACTORS
            'Market_Cap': 'Size',
            'EV': 'Size',
            'Revenue': 'Size',
            'FwdRevenue': 'Size',
            'Price': 'Size',
            'Assets': 'Size',
            
            # RISK FACTORS
            'RollingVol': 'Risk',
            'Ten_Year': 'Risk',
            
            # SENTIMENT FACTORS
            'SI': 'Sentiment',
            'SI_Ratio': 'Sentiment',
            
            # MACRO FACTORS
            'GDP': 'Macro',
            'M2': 'Macro'
        }
        
        
        classification_map={}
        for key in factor_category.keys():
            #1mo
            classification_map[key +'_1mo_pct_chg'] = factor_category[key]
            classification_map[key +'_1mo_diff_chg'] = factor_category[key]
            
            #3mo
            classification_map[key +'_3mo_pct_chg'] = factor_category[key]
            classification_map[key +'_3mo_diff_chg'] = factor_category[key]            
            
            #6mo
            classification_map[key +'_6mo_pct_chg'] = factor_category[key]
            classification_map[key +'_6mo_diff_chg'] = factor_category[key]

            #12mo
            classification_map[key +'_12mo_pct_chg'] = factor_category[key]
            classification_map[key +'_12mo_diff_chg'] = factor_category[key]                        
            
            #Absolute
            classification_map[key] = factor_category[key]
        
        return classification_map
    
    def create_signal_directionality_map(self) -> Dict[str, int]:
        """
        Create a signal directionality map where 1 means "higher is better" and -1 means "lower is better".
        
        This ensures all factors are oriented consistently: higher values = more attractive investment.
        
        Returns:
        --------
        Dict[str, int] : Factor to directionality mapping (1 or -1)
        """
        
        direction = {
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
            
            # YIELD FACTORS (Higher yields = better, so KEEP)
            'EarningsYieldTTM': 1,      # Higher earnings yield is better
            'EarningsYieldFWD': 1,      # Higher forward earnings yield is better
            'CashFlowYieldTTM': 1,      # Higher cash flow yield is better
            'CashFlowYieldFWD': 1,      # Higher forward cash flow yield is better
            
            # YIELD SPREADS (Higher spreads = better compensation, so KEEP)
            'EarningsYieldTTMSpread': 1,
            'EarningsYieldFWDSpread': 1,
            'CashFlowYieldTTMSpread': 1,
            'CashFlowYieldFWDSpread': 1,
            'DvdYieldTTMSpread': 1,
            'DvdYieldFWDSpread': 1,
            
            # QUALITY FACTORS (Mixed - depends on specific metric)
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
            
            # PROFITABILITY FACTORS (Higher margins/profits = better, so KEEP)
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
            
            # MOMENTUM FACTORS (Higher growth = better, so KEEP)
            'ConsensusSalesGrowth': 1,      # Higher sales growth is better
            'ConsensusEbitdaGrowth': 1,     # Higher EBITDA growth is better
            'ConsensusEarningsGrowth': 1,   # Higher earnings growth is better
            'ConsensusCashFlowGrowth': 1,   # Higher cash flow growth is better
            'RollingEarnings': 1,           # Higher rolling earnings growth is better
            'FwdRollingEarnings': 1,        # Higher forward rolling earnings is better
            'CumFlow': 1,                   # Positive cumulative flows are better
            'Flows': 1,                     # Positive flows are better
            
            # SIZE FACTORS (Depends on investment philosophy - assume larger is better for liquidity)
            'Market_Cap': 1,            # Larger market cap (more liquidity, stability)
            'EV': 1,                    # Larger enterprise value
            'Revenue': 1,               # Higher revenue (scale)
            'FwdRevenue': 1,            # Higher forward revenue
            'Price': 1,                 # Price itself is neutral, but momentum positive
            
            # RISK FACTORS (Lower risk = better, so INVERT)
            'RollingVol': -1,           # Lower volatility is better (less risky)
            'Ten_Year': 1,              # Higher bond yields could mean higher risk premiums
            
            # SENTIMENT FACTORS (High short interest = negative sentiment, so INVERT)
            'SI': -1,                   # Lower short interest is better
            'SI_Ratio': -1,             # Lower short interest ratio is better
            
            # MACRO FACTORS (Higher growth = better, so KEEP)
            'GDP': 1,                   # Higher GDP growth is better
            'M2': 1,                    # Money supply growth can be positive for assets
        }

        directionality_map={}
        for key in direction.keys():
            #1mo
            directionality_map[key +'_1mo_pct_chg'] = direction[key]
            directionality_map[key +'_1mo_diff_chg'] = direction[key]
            
            #3mo
            directionality_map[key +'_3mo_pct_chg'] = direction[key]
            directionality_map[key +'_3mo_diff_chg'] = direction[key]            
            
            #6mo
            directionality_map[key +'_6mo_pct_chg'] = direction[key]
            directionality_map[key +'_6mo_diff_chg'] = direction[key]

            #12mo
            directionality_map[key +'_12mo_pct_chg'] = direction[key]
            directionality_map[key +'_12mo_diff_chg'] = direction[key]                        
            
            #Absolute
            directionality_map[key] = direction[key]
        
        return directionality_map
    
    def apply_signal_directionality(self, dataFrames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply signal directionality corrections to ensure all factors point in the same direction.
        
        Parameters:
        -----------
        dataFrames : Dict[str, pd.DataFrame]
            Original factor data
            
        Returns:
        --------
        Dict[str, pd.DataFrame] : Directionally corrected factor data
        """
        print("Applying signal directionality corrections...")
        
        directionality_map = self.create_signal_directionality_map()
        corrected_dataFrames = {}
        correction_summary = {'inverted': [], 'kept_as_is': [], 'unknown_factors': []}
        
        for factor_name, factor_df in dataFrames.items():
            if factor_name in directionality_map:
                direction = directionality_map[factor_name]
                
                if direction == -1:
                    # Invert the factor (multiply by -1)
                    corrected_dataFrames[factor_name] = -factor_df
                    correction_summary['inverted'].append(factor_name)
                    print(f"  ✓ Inverted {factor_name} (lower was better)")
                else:
                    # Keep as-is (higher is better)
                    corrected_dataFrames[factor_name] = factor_df.copy()
                    correction_summary['kept_as_is'].append(factor_name)
                    print(f"  ✓ Kept {factor_name} as-is (higher is better)")
            else:
                # Unknown factor - keep as-is but flag for review
                corrected_dataFrames[factor_name] = factor_df.copy()
                correction_summary['unknown_factors'].append(factor_name)
                print(f"  ⚠️  Unknown factor {factor_name} - kept as-is (please review directionality)")
        
        # Store correction summary
        self.results['signal_corrections'] = correction_summary
        
        print(f"\nSignal Directionality Summary:")
        print(f"  Factors inverted: {len(correction_summary['inverted'])}")
        print(f"  Factors kept as-is: {len(correction_summary['kept_as_is'])}")
        print(f"  Unknown factors: {len(correction_summary['unknown_factors'])}")
        
        if correction_summary['unknown_factors']:
            print(f"  ⚠️  Please review directionality for: {correction_summary['unknown_factors']}")
        
        return corrected_dataFrames
    
    def create_country_factor_matrix(self, dataFrames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a standardized factor matrix where each row is a country-date observation
        and each column is a standardized factor.
        
        Parameters:
        -----------
        dataFrames : Dict[str, pd.DataFrame]
            Dictionary where keys are metric names and values are DataFrames with
            countries as columns and dates as index
            
        Returns:
        --------
        pd.DataFrame : Standardized country-factor matrix
        """
        from sklearn.preprocessing import StandardScaler
        
        print("Creating country-factor matrix...")
        print(f"Processing {len(dataFrames)} metrics...")
        
        if not dataFrames:
            raise ValueError("No metrics provided in dataFrames dictionary")
        
        # Step 1: Stack all DataFrames to create long-format data
        factor_panels = {}
        
        for metric_name, metric_df in dataFrames.items():
            if isinstance(metric_df, pd.DataFrame) and not metric_df.empty:
                print(f"  Processing {metric_name}: shape {metric_df.shape}")
                
                try:
                    # Stack the DataFrame: (date, country) -> value
                    stacked = metric_df.stack()
                    stacked.name = metric_name
                    
                    # Check data coverage
                    total_possible = len(metric_df.index) * len(metric_df.columns)
                    coverage = stacked.notna().sum() / total_possible if total_possible > 0 else 0
                    
                    if coverage >= self.min_data_coverage:
                        factor_panels[metric_name] = stacked
                        print(f"    ✓ {metric_name}: {coverage:.1%} coverage, {stacked.notna().sum()} valid observations")
                    else:
                        print(f"    ✗ {metric_name}: {coverage:.1%} coverage (below {self.min_data_coverage:.1%} threshold)")
                        
                except Exception as e:
                    print(f"    ✗ Error processing {metric_name}: {e}")
            else:
                print(f"  ✗ Skipping {metric_name}: empty or invalid DataFrame")
        
        # Step 2: Combine all factors into single DataFrame
        print(f"\nCombining {len(factor_panels)} qualifying metrics...")
        
        if not factor_panels:
            raise ValueError("No metrics meet the minimum data coverage requirement")
        
        # Create multi-index DataFrame
        combined_data = pd.DataFrame(factor_panels)
        print(f"Combined data shape: {combined_data.shape}")
        print(f"Non-null values in combined data: {combined_data.notna().sum().sum()}")
        
        if combined_data.empty:
            raise ValueError("Combined data is empty after stacking")
        
        # Step 3: Cross-sectional standardization at each date using StandardScaler
        print("Performing cross-sectional standardization...")
        
        standardized_data = pd.DataFrame(index=combined_data.index, 
                                       columns=combined_data.columns,
                                       dtype=float)
        
        # Check if we have MultiIndex (date, country) structure
        if isinstance(combined_data.index, pd.MultiIndex):
            print("Processing MultiIndex structure...")
            
            # Get unique dates from first level
            unique_dates = combined_data.index.get_level_values(0).unique()
            print(f"Found {len(unique_dates)} unique dates")
            
            dates_processed = 0
            dates_with_data = 0
            
            for date in unique_dates:
                try:
                    # Get data for this date across all countries
                    date_data = combined_data.loc[date]
                    
                    if isinstance(date_data, pd.Series):
                        # Only one country for this date, convert to DataFrame
                        date_data = date_data.to_frame().T
                    
                    if len(date_data) > 0 and not date_data.empty:
                        # Apply StandardScaler column by column (factor by factor)
                        for col in date_data.columns:
                            col_values = date_data[col].dropna()
                            
                            if len(col_values) >= 2:  # Need at least 2 countries for standardization
                                # Use StandardScaler for robust standardization
                                scaler = StandardScaler()
                                try:
                                    # Reshape for sklearn
                                    values_reshaped = col_values.values.reshape(-1, 1)
                                    standardized_values = scaler.fit_transform(values_reshaped).flatten()
                                    
                                    # Map back to original index
                                    for idx, std_val in zip(col_values.index, standardized_values):
                                        standardized_data.loc[(date, idx), col] = std_val
                                        
                                    dates_with_data += 1
                                    
                                except Exception as e:
                                    print(f"      Warning: StandardScaler failed for {col} on {date}: {e}")
                                    # Fallback to manual standardization
                                    mean_val = col_values.mean()
                                    std_val = col_values.std()
                                    if std_val > 0:
                                        standardized_vals = (col_values - mean_val) / std_val
                                        for idx, std_val in zip(col_values.index, standardized_vals):
                                            standardized_data.loc[(date, idx), col] = std_val
                            
                            elif len(col_values) == 1:
                                # Only one country - set to zero (neutral)
                                standardized_data.loc[(date, col_values.index[0]), col] = 0.0
                    
                    dates_processed += 1
                    if dates_processed % 100 == 0:
                        print(f"    Processed {dates_processed}/{len(unique_dates)} dates, {dates_with_data} with valid data")
                        
                except Exception as e:
                    print(f"    Warning: Error processing date {date}: {e}")
                    continue
            
            print(f"✓ Processed {dates_processed} dates, {dates_with_data} dates had valid data")
            
        else:
            # Regular index - assume each row is an observation
            print("Processing regular index structure...")
            print("Warning: Expected MultiIndex structure, attempting simple standardization")
            
            # Apply StandardScaler to each column independently
            for col in combined_data.columns:
                col_data = combined_data[col].dropna()
                
                if len(col_data) > 1:
                    scaler = StandardScaler()
                    try:
                        values_reshaped = col_data.values.reshape(-1, 1)
                        standardized_values = scaler.fit_transform(values_reshaped).flatten()
                        
                        # Map back to original positions
                        for idx, std_val in zip(col_data.index, standardized_values):
                            standardized_data.loc[idx, col] = std_val
                            
                    except Exception as e:
                        print(f"    Warning: StandardScaler failed for {col}: {e}")
                        # Fallback
                        mean_val = col_data.mean()
                        std_val = col_data.std()
                        if std_val > 0:
                            standardized_data[col] = (combined_data[col] - mean_val) / std_val
                        else:
                            standardized_data[col] = combined_data[col] - mean_val
                else:
                    # Insufficient data
                    standardized_data[col] = combined_data[col]
        
        # Step 4: Clean up the standardized data and validate
        print("Cleaning up standardized data...")
        
        # Remove rows and columns that are entirely NaN
        final_data = standardized_data.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        print(f"Data cleanup: {standardized_data.shape} → {final_data.shape}")
        print(f"Non-null values after standardization: {final_data.notna().sum().sum()}")
        
        if final_data.empty:
            print("ERROR: Final standardized matrix is empty!")
            print("Debug information:")
            print(f"  - Original combined data shape: {combined_data.shape}")
            print(f"  - Non-null values in combined data: {combined_data.notna().sum().sum()}")
            print(f"  - Standardized data shape before cleanup: {standardized_data.shape}")
            print(f"  - Non-null values in standardized data: {standardized_data.notna().sum().sum()}")
            
            # Debug: Show sample of what we have
            print(f"  - Sample combined data:")
            print(combined_data.head())
            print(f"  - Sample standardized data:")
            print(standardized_data.head())
            
            # Return the combined data as fallback
            print("  - Returning unstandardized combined data as fallback")
            final_data = combined_data.dropna(how='all', axis=0).dropna(how='all', axis=1)
            
        else:
            # Validate standardization worked
            print("Standardization validation:")
            for col in final_data.columns:
                col_data = final_data[col].dropna()
                if len(col_data) > 1:
                    print(f"  {col}: mean={col_data.mean():.3f}, std={col_data.std():.3f}")
        
        print(f"✓ Created factor matrix: {final_data.shape[0]} observations × {final_data.shape[1]} factors")
        
        # Store metadata
        self.results['data_info'] = {
            'original_metrics': len(dataFrames),
            'qualifying_metrics': len(factor_panels),
            'final_observations': final_data.shape[0],
            'final_factors': final_data.shape[1],
            'coverage_stats': {metric: factor_panels[metric].notna().mean() 
                             for metric in factor_panels.keys()}
        }
        
        # Store intermediate results for debugging
        self.results['combined_data'] = combined_data
        self.results['standardized_data'] = standardized_data
        self.results['final_data'] = final_data
        
        return final_data
    
    def calculate_correlation_matrix(self, factor_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix across all country-date observations.
        
        Parameters:
        -----------
        factor_matrix : pd.DataFrame
            Standardized country-factor matrix
            
        Returns:
        --------
        pd.DataFrame : Factor correlation matrix
        """
        print("Computing factor correlation matrix...")
        
        # Remove any remaining NaN values
        clean_data = factor_matrix.dropna()
        print(f"  Using {len(clean_data)} complete observations")
        
        if factor_matrix.empty:
           print("WARNING: Factor matrix is empty, returning empty correlation matrix")
           return pd.DataFrame()
        
        # Remove any remaining NaN values
        clean_data = factor_matrix.dropna()
        print(f"  Using {len(clean_data)} complete observations")
        
        if len(clean_data) == 0:
            print("WARNING: No complete observations after dropna()")
            return pd.DataFrame()
        
        # Calculate Spearman correlation (robust to outliers)
        correlation_matrix = clean_data.corr(method='spearman')
        
        # Calculate statistical significance
        n_obs = len(clean_data)
        p_values = pd.DataFrame(index=correlation_matrix.index, 
                               columns=correlation_matrix.columns,
                               dtype=float)
        
        for i, factor1 in enumerate(correlation_matrix.columns):
            for j, factor2 in enumerate(correlation_matrix.columns):
                if i <= j:
                    if i == j:
                        p_values.loc[factor1, factor2] = 0.0
                    else:
                        corr_val = correlation_matrix.loc[factor1, factor2]
                        if not np.isnan(corr_val) and abs(corr_val) < 0.999:
                            # T-test for correlation significance
                            t_stat = corr_val * np.sqrt((n_obs-2) / (1 - corr_val**2))
                            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), n_obs-2))
                            p_values.loc[factor1, factor2] = p_val
                            p_values.loc[factor2, factor1] = p_val
                        else:
                            p_values.loc[factor1, factor2] = 0.0
                            p_values.loc[factor2, factor1] = 0.0
        
        # Identify highly correlated pairs
        high_corr_mask = (np.abs(correlation_matrix) >= self.correlation_threshold) & (p_values < 0.05)
        high_corr_pairs = []
        
        for i, factor1 in enumerate(correlation_matrix.columns):
            for j, factor2 in enumerate(correlation_matrix.columns):
                if i < j and high_corr_mask.loc[factor1, factor2]:
                    high_corr_pairs.append({
                        'factor1': factor1,
                        'factor2': factor2,
                        'correlation': correlation_matrix.loc[factor1, factor2],
                        'p_value': p_values.loc[factor1, factor2]
                    })
        
        self.results['high_correlations'] = high_corr_pairs
        print(f"✓ Found {len(high_corr_pairs)} highly correlated pairs (|r| >= {self.correlation_threshold})")
        
        return correlation_matrix
    
    def hierarchical_clustering_selection(self, correlation_matrix: pd.DataFrame, 
                                        classification_map: Dict[str, str]) -> List[str]:
        """
        Use hierarchical clustering to select representative factors from each cluster.
        
        Parameters:
        -----------
        correlation_matrix : pd.DataFrame
            Factor correlation matrix
        classification_map : Dict[str, str]
            Factor category mapping
            
        Returns:
        --------
        List[str] : Selected representative factors
        """
        print("Applying hierarchical clustering for factor selection...")
        
        # Convert correlation to distance
        distance_matrix = 1 - np.abs(correlation_matrix.fillna(0))
        np.fill_diagonal(distance_matrix.values, 0)
        
        # Hierarchical clustering
        try:
            linkage_matrix = linkage(distance_matrix, method='ward')
            
            # Determine clusters
            cluster_threshold = 1 - self.correlation_threshold
            cluster_labels = fcluster(linkage_matrix, cluster_threshold, criterion='distance')
            
        except Exception as e:
            print(f"  Warning: Clustering failed ({e}), using correlation-based grouping")
            # Fallback: simple correlation-based grouping
            cluster_labels = self._correlation_based_clustering(correlation_matrix)
        
        # Analyze clusters
        cluster_df = pd.DataFrame({
            'factor': correlation_matrix.index,
            'cluster': cluster_labels,
            'category': [classification_map.get(f, 'Unknown') for f in correlation_matrix.index]
        })
        
        selected_factors = []
        cluster_analysis = []
        
        for cluster_id in np.unique(cluster_labels):
            cluster_factors = cluster_df[cluster_df['cluster'] == cluster_id]['factor'].tolist()
            
            if len(cluster_factors) == 1:
                # Single factor cluster
                selected_factors.extend(cluster_factors)
                cluster_analysis.append({
                    'cluster_id': cluster_id,
                    'size': 1,
                    'factors': cluster_factors,
                    'selected': cluster_factors[0],
                    'selection_method': 'single_factor'
                })
            else:
                # Multi-factor cluster - select representative
                cluster_corr_matrix = correlation_matrix.loc[cluster_factors, cluster_factors]
                
                # Method 1: Factor with highest average absolute correlation within cluster
                avg_abs_corr = cluster_corr_matrix.abs().mean(axis=1)
                representative = avg_abs_corr.idxmax()
                
                # Method 2: Prefer factors from underrepresented categories
                cluster_categories = [classification_map.get(f, 'Unknown') for f in cluster_factors]
                category_counts = pd.Series(cluster_categories).value_counts()
                
                # If we have category diversity, prefer less common categories
                if len(category_counts) > 1:
                    rarest_category = category_counts.idxmin()
                    rare_factors = [f for f in cluster_factors 
                                  if classification_map.get(f, 'Unknown') == rarest_category]
                    if rare_factors:
                        # Among rare category factors, pick the one with highest avg correlation
                        rare_corr = cluster_corr_matrix.loc[rare_factors].abs().mean(axis=1)
                        representative = rare_corr.idxmax()
                
                selected_factors.append(representative)
                cluster_analysis.append({
                    'cluster_id': cluster_id,
                    'size': len(cluster_factors),
                    'factors': cluster_factors,
                    'selected': representative,
                    'selection_method': 'representative_selection',
                    'avg_correlation': cluster_corr_matrix.abs().mean().mean()
                })
        
        self.results['clustering_analysis'] = cluster_analysis
        print(f"✓ Clustering reduced {len(correlation_matrix.columns)} factors to {len(selected_factors)}")
        
        return selected_factors
    
    def _correlation_based_clustering(self, correlation_matrix: pd.DataFrame) -> np.ndarray:
        """Fallback clustering method based on correlation threshold."""
        
        n_factors = len(correlation_matrix)
        cluster_labels = np.arange(n_factors)  # Start with each factor in its own cluster
        
        # Find highly correlated pairs and merge clusters
        high_corr_pairs = []
        for i in range(n_factors):
            for j in range(i+1, n_factors):
                if abs(correlation_matrix.iloc[i, j]) >= self.correlation_threshold:
                    high_corr_pairs.append((i, j, abs(correlation_matrix.iloc[i, j])))
        
        # Sort by correlation strength
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Merge clusters
        for i, j, corr in high_corr_pairs:
            if cluster_labels[i] != cluster_labels[j]:
                # Merge cluster j into cluster i
                old_label = cluster_labels[j]
                cluster_labels[cluster_labels == old_label] = cluster_labels[i]
        
        # Renumber clusters to be consecutive
        unique_labels = np.unique(cluster_labels)
        for new_id, old_id in enumerate(unique_labels, 1):
            cluster_labels[cluster_labels == old_id] = new_id
        
        return cluster_labels
    
    def variance_inflation_factor_analysis(self, factor_matrix: pd.DataFrame, 
                                         candidate_factors: List[str]) -> List[str]:
        """
        Apply VIF analysis to remove multicollinear factors.
        
        Parameters:
        -----------
        factor_matrix : pd.DataFrame
            Standardized factor matrix
        candidate_factors : List[str]
            Factors to analyze
            
        Returns:
        --------
        List[str] : Factors passing VIF test
        """
        print("Performing VIF analysis...")
        
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            print("  ⚠️  statsmodels not available, skipping VIF analysis")
            return candidate_factors
        
        # Prepare clean data
        vif_data = factor_matrix[candidate_factors].dropna()
        
        if len(vif_data) < len(candidate_factors) * 5:
            print("  ⚠️  Insufficient data for VIF analysis")
            return candidate_factors
        
        remaining_factors = candidate_factors.copy()
        vif_results = []
        iteration = 1
        
        while len(remaining_factors) > 1:
            print(f"  VIF iteration {iteration}: testing {len(remaining_factors)} factors")
            
            current_data = vif_data[remaining_factors].values
            current_vifs = {}
            
            # Calculate VIF for each factor
            for i, factor in enumerate(remaining_factors):
                try:
                    vif_score = variance_inflation_factor(current_data, i)
                    if np.isfinite(vif_score):
                        current_vifs[factor] = vif_score
                    else:
                        current_vifs[factor] = 1.0
                except:
                    current_vifs[factor] = 1.0
            
            # Find highest VIF
            max_vif_factor = max(current_vifs, key=current_vifs.get)
            max_vif_value = current_vifs[max_vif_factor]
            
            vif_results.append({
                'iteration': iteration,
                'factor': max_vif_factor,
                'vif': max_vif_value,
                'action': 'removed' if max_vif_value > self.vif_threshold else 'kept'
            })
            
            if max_vif_value > self.vif_threshold:
                remaining_factors.remove(max_vif_factor)
                print(f"    Removed {max_vif_factor} (VIF: {max_vif_value:.2f})")
                iteration += 1
            else:
                break
        
        self.results['vif_analysis'] = vif_results
        self.results['remaining_factors_post_vif'] = remaining_factors
        print(f"✓ VIF analysis: {len(candidate_factors)} → {len(remaining_factors)} factors")
        
        return remaining_factors
    
    def balance_factor_categories(self, selected_factors: List[str], 
                                classification_map: Dict[str, str],
                                max_per_category: int = 4) -> List[str]:
        """
        Balance selected factors across categories.
        
        Parameters:
        -----------
        selected_factors : List[str]
            Previously selected factors
        classification_map : Dict[str, str]
            Factor classification mapping
        max_per_category : int, default 4
            Maximum factors per category
            
        Returns:
        --------
        List[str] : Category-balanced factors
        """
        print("Balancing factors across categories...")
        
        # Group by category
        category_groups = {}
        for factor in selected_factors:
            category = classification_map.get(factor, 'Unknown')
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(factor)
        
        balanced_factors = []
        balancing_info = {}
        
        for category, factors in category_groups.items():
            original_count = len(factors)
            
            if original_count <= max_per_category:
                balanced_factors.extend(factors)
                final_count = original_count
            else:
                # Keep first max_per_category factors
                # TODO: Could enhance with better selection criteria
                selected_subset = factors[:max_per_category]
                balanced_factors.extend(selected_subset)
                final_count = max_per_category
                print(f"  Limited {category}: {original_count} → {final_count} factors")
            
            balancing_info[category] = {
                'original': original_count,
                'final': final_count,
                'factors': factors[:final_count] if original_count > max_per_category else factors
            }
        
        self.results['category_balancing'] = balancing_info
        print(f"✓ Category balancing: {len(selected_factors)} → {len(balanced_factors)} factors")
        
        return balanced_factors
    
    def run_complete_analysis(self, dataFrames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run the complete country-level factor selection analysis with signal directionality correction.
        
        Parameters:
        -----------
        dataFrames : Dict[str, pd.DataFrame]
            Dictionary with metric names as keys and country time-series as values
            
        Returns:
        --------
        Dict[str, Any] : Complete analysis results
        """
        print("="*70)
        print("COUNTRY-LEVEL QUANTITATIVE FACTOR SELECTION ANALYSIS")
        print("WITH SIGNAL DIRECTIONALITY CORRECTION")
        print("="*70)
        
        # Step 0: Apply signal directionality corrections
        corrected_dataFrames = self.apply_signal_directionality(dataFrames)
        
        # Step 1: Create classification map
        classification_map = self.create_classification_map()
        
        # Step 2: Create and standardize country-factor matrix (using corrected data)
        factor_matrix = self.create_country_factor_matrix(corrected_dataFrames)
        
        # Step 3: Calculate correlation matrix
        correlation_matrix = self.calculate_correlation_matrix(factor_matrix)
        
        # Step 4: Hierarchical clustering selection
        clustered_factors = self.hierarchical_clustering_selection(correlation_matrix, classification_map)
        
        # Step 5: VIF analysis
        vif_filtered_factors = self.variance_inflation_factor_analysis(factor_matrix, clustered_factors)
        
        # Step 6: Category balancing
        final_factors = self.balance_factor_categories(vif_filtered_factors, classification_map)
        
        # Compile results
        results = {
            'classification_map': classification_map,
            'directionality_map': self.create_signal_directionality_map(),
            'original_dataframes': dataFrames,
            'corrected_dataframes': corrected_dataFrames,
            'factor_matrix': factor_matrix,
            'correlation_matrix': correlation_matrix,
            'original_factor_count': len(dataFrames),
            'clustered_factors': clustered_factors,
            'vif_filtered_factors': vif_filtered_factors,
            'final_factors': final_factors,
            'final_factor_count': len(final_factors),
            'reduction_ratio': len(final_factors) / len(dataFrames),
            'processing_details': self.results
        }
        
        # Generate comprehensive summary
        self._generate_comprehensive_summary(results)
        
        return results
    
    def _generate_comprehensive_summary(self, results: Dict[str, Any]) -> None:
        """Generate detailed analysis summary."""
        
        print("\n" + "="*70)
        print("FACTOR SELECTION SUMMARY")
        print("="*70)
        
        # Overall statistics
        print(f"Original metrics: {results['original_factor_count']}")
        print(f"Selected factors: {results['final_factor_count']}")
        print(f"Reduction ratio: {results['reduction_ratio']:.1%}")
        
        # Signal directionality summary
        if 'signal_corrections' in self.results:
            corrections = self.results['signal_corrections']
            print(f"\nSignal Directionality Corrections:")
            print(f"  Factors inverted: {len(corrections['inverted'])}")
            print(f"  Factors kept as-is: {len(corrections['kept_as_is'])}")
            print(f"  Unknown factors: {len(corrections['unknown_factors'])}")
            
            if corrections['inverted']:
                print(f"  Inverted factors: {', '.join(corrections['inverted'][:5])}" + 
                     (f" ... (+{len(corrections['inverted'])-5} more)" if len(corrections['inverted']) > 5 else ""))
            
            if corrections['unknown_factors']:
                print(f"  ⚠️  Review needed: {', '.join(corrections['unknown_factors'])}")
        
        # Data coverage info
        if 'data_info' in self.results:
            info = self.results['data_info']
            print(f"\nData Quality:")
            print(f"  Qualifying metrics: {info['qualifying_metrics']}/{info['original_metrics']}")
            print(f"  Total observations: {info['final_observations']:,}")
            print(f"  Average coverage: {np.mean(list(info['coverage_stats'].values())):.1%}")
        
        # Category breakdown
        classification_map = results['classification_map']
        final_factors = results['final_factors']
        
        print(f"\nFactor Distribution by Category:")
        category_counts = {}
        for factor in final_factors:
            category = classification_map.get(factor, 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category, count in sorted(category_counts.items()):
            print(f"  {category:<15}: {count} factors")
        
        # High correlation summary
        if 'high_correlations' in self.results:
            n_high_corr = len(self.results['high_correlations'])
            print(f"\nCorrelation Analysis:")
            print(f"  High correlations found: {n_high_corr}")
            print(f"  Correlation threshold: {self.correlation_threshold}")
        
        # Clustering summary
        if 'clustering_analysis' in self.results:
            clusters = self.results['clustering_analysis']
            multi_factor_clusters = [c for c in clusters if c['size'] > 1]
            print(f"\nClustering Results:")
            print(f"  Total clusters: {len(clusters)}")
            print(f"  Multi-factor clusters: {len(multi_factor_clusters)}")
            
            if multi_factor_clusters:
                avg_cluster_size = np.mean([c['size'] for c in multi_factor_clusters])
                print(f"  Average cluster size: {avg_cluster_size:.1f}")
        
        # Final factor list
        print(f"\nSelected Factors ({len(final_factors)}):")
        print("-" * 50)
        
        # Group by category for display
        categorized_factors = {}
        directionality_map = results['directionality_map']
        
        for factor in sorted(final_factors):
            category = classification_map.get(factor, 'Unknown')
            if category not in categorized_factors:
                categorized_factors[category] = []
            
            # Add directionality info
            direction = directionality_map.get(factor, 1)
            direction_symbol = "↑" if direction == 1 else "↓" if direction == -1 else "?"
            
            categorized_factors[category].append(f"{factor} {direction_symbol}")
        
        for category in sorted(categorized_factors.keys()):
            print(f"\n{category}:")
            for i, factor_info in enumerate(categorized_factors[category], 1):
                print(f"  {i}. {factor_info}")
        
        print(f"\nLegend: ↑ = Higher is better, ↓ = Inverted (lower was better)")
    
    def create_final_factor_matrix(self, dataFrames: Dict[str, pd.DataFrame], 
                                 selected_factors: List[str]) -> pd.DataFrame:
        """
        Create final standardized factor matrix with only selected factors and proper directionality.
        
        Parameters:
        -----------
        dataFrames : Dict[str, pd.DataFrame]
            Original data dictionary
        selected_factors : List[str]
            Final selected factors
            
        Returns:
        --------
        pd.DataFrame : Final factor matrix for modeling
        """
        print(f"Creating final factor matrix with {len(selected_factors)} factors...")
        
        # Apply directionality corrections first
        corrected_data = self.apply_signal_directionality(dataFrames)
        
        # Filter to selected factors only
        selected_data = {factor: corrected_data[factor] for factor in selected_factors 
                        if factor in corrected_data}
        
        # Create standardized matrix using same process
        final_matrix = self.create_country_factor_matrix(selected_data)
        
        print(f"✓ Final matrix shape: {final_matrix.shape}")
        return final_matrix
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                               selected_factors: List[str] = None,
                               figsize: Tuple[int, int] = (14, 12)) -> None:
        """
        Plot correlation heatmap for analysis with directionality annotations.
        
        Parameters:
        -----------
        correlation_matrix : pd.DataFrame
            Factor correlation matrix
        selected_factors : List[str], optional
            Factors to highlight, if None uses all
        figsize : Tuple[int, int]
            Figure size
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if selected_factors:
            plot_matrix = correlation_matrix.loc[selected_factors, selected_factors]
            title = f'Correlation Matrix - Selected Factors ({len(selected_factors)})'
        else:
            plot_matrix = correlation_matrix
            title = f'Correlation Matrix - All Factors ({len(correlation_matrix)})'
        
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(plot_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(plot_matrix, 
                   mask=mask,
                   annot=True if len(plot_matrix) <= 20 else False,
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        # Add directionality annotations if available
        if hasattr(self, 'results') and 'signal_corrections' in self.results:
            corrections = self.results['signal_corrections']
            
            # Create legend text
            legend_text = []
            if corrections['inverted']:
                legend_text.append(f"Inverted factors ({len(corrections['inverted'])}): " + 
                                 ", ".join(corrections['inverted'][:3]) + 
                                 ("..." if len(corrections['inverted']) > 3 else ""))
            
            if legend_text:
                plt.figtext(0.02, 0.02, "\n".join(legend_text), 
                           fontsize=8, style='italic', wrap=True)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Factors', fontsize=12)
        plt.ylabel('Factors', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
    def get_directionality_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive directionality report for all factors.
        
        Returns:
        --------
        pd.DataFrame : Directionality report
        """
        directionality_map = self.create_signal_directionality_map()
        classification_map = self.create_classification_map()
        
        report_data = []
        for factor, direction in directionality_map.items():
            report_data.append({
                'Factor': factor,
                'Category': classification_map.get(factor, 'Unknown'),
                'Direction': direction,
                'Interpretation': 'Higher is better' if direction == 1 else 'Lower is better (inverted)',
                'Action': 'Keep as-is' if direction == 1 else 'Multiply by -1'
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values(['Category', 'Factor'])
        
        return report_df


def create_factor_analysis_report(results: Dict[str, Any], 
                               output_file: str = 'factor_selection_report.txt') -> None:
   """
   Create a comprehensive text report of the factor selection analysis.
   
   Parameters:
   -----------
   results : Dict[str, Any]
       Results from run_complete_analysis
   output_file : str
       Output file path
   """
   
   with open(output_file, 'w', encoding='utf-8') as f:
       f.write("="*80 + "\n")
       f.write("QUANTITATIVE FACTOR SELECTION ANALYSIS REPORT\n")
       f.write("WITH SIGNAL DIRECTIONALITY CORRECTION\n")
       f.write("="*80 + "\n\n")
       
       if 'error' in results:
           f.write("ANALYSIS FAILED\n")
           f.write("-"*40 + "\n")
           f.write(f"Error: {results['error']}\n\n")
           return
       
       # Executive Summary
       f.write("EXECUTIVE SUMMARY\n")
       f.write("-"*40 + "\n")
       f.write(f"Original Metrics: {results['original_factor_count']}\n")
       f.write(f"Selected Factors: {results['final_factor_count']}\n")
       f.write(f"Reduction Ratio: {results['reduction_ratio']:.1%}\n\n")
       
       # Signal Directionality Summary
       if 'signal_corrections' in results['processing_details']:
           corrections = results['processing_details']['signal_corrections']
           f.write("SIGNAL DIRECTIONALITY CORRECTIONS\n")
           f.write("-"*40 + "\n")
           f.write(f"Factors Inverted: {len(corrections['inverted'])}\n")
           f.write(f"Factors Kept As-Is: {len(corrections['kept_as_is'])}\n")
           f.write(f"Unknown Factors: {len(corrections['unknown_factors'])}\n")
           
           if corrections['inverted']:
               f.write(f"\nInverted Factors (Lower was better):\n")
               for i, factor in enumerate(corrections['inverted'], 1):
                   f.write(f"  {i:2d}. {factor}\n")
           
           if corrections['unknown_factors']:
               f.write(f"\nUnknown Factors (Need Manual Review):\n")
               for i, factor in enumerate(corrections['unknown_factors'], 1):
                   f.write(f"  {i:2d}. {factor}\n")
           f.write("\n")
       
       # Data Quality
       if 'data_info' in results['processing_details']:
           info = results['processing_details']['data_info']
           f.write("DATA QUALITY ASSESSMENT\n")
           f.write("-"*40 + "\n")
           f.write(f"Qualifying Metrics: {info['qualifying_metrics']}/{info['original_metrics']}\n")
           f.write(f"Total Observations: {info['final_observations']:,}\n")
           if info['coverage_stats']:
               f.write(f"Average Coverage: {np.mean(list(info['coverage_stats'].values())):.1%}\n\n")
       
       # Selected Factors by Category (with directionality)
       if results['final_factors']:
           classification_map = results['classification_map']
           directionality_map = results['directionality_map']
           final_factors = results['final_factors']
           
           categorized_factors = {}
           for factor in sorted(final_factors):
               category = classification_map.get(factor, 'Unknown')
               if category not in categorized_factors:
                   categorized_factors[category] = []
               
               # Add directionality symbol
               direction = directionality_map.get(factor, 1)
               direction_symbol = " ↑" if direction == 1 else " ↓" if direction == -1 else " ?"
               categorized_factors[category].append(factor + direction_symbol)
           
           f.write("SELECTED FACTORS BY CATEGORY\n")
           f.write("-"*40 + "\n")
           f.write("Legend: ↑ = Higher is better, ↓ = Inverted (lower was better)\n\n")
           
           for category in sorted(categorized_factors.keys()):
               f.write(f"{category.upper()} ({len(categorized_factors[category])}):\n")
               for i, factor_info in enumerate(categorized_factors[category], 1):
                   f.write(f"  {i:2d}. {factor_info}\n")
               f.write("\n")
       
       # Correlation Analysis
       if 'high_correlations' in results['processing_details']:
           high_corrs = results['processing_details']['high_correlations']
           f.write(f"CORRELATION ANALYSIS\n")
           f.write("-"*40 + "\n")
           f.write(f"High Correlations Identified: {len(high_corrs)}\n")
           f.write(f"Note: Correlations calculated AFTER directionality corrections\n")
           
           if high_corrs:
               f.write("\nTop 10 Highest Correlations (before clustering):\n")
               sorted_corrs = sorted(high_corrs, key=lambda x: abs(x['correlation']), reverse=True)
               for i, corr in enumerate(sorted_corrs[:10], 1):
                   f.write(f"  {i:2d}. {corr['factor1']} <-> {corr['factor2']}: "
                          f"{corr['correlation']:+.3f} (p={corr['p_value']:.3f})\n")
           f.write("\n")
       
       # Clustering Analysis
       if 'clustering_analysis' in results['processing_details']:
           clusters = results['processing_details']['clustering_analysis']
           multi_clusters = [c for c in clusters if c['size'] > 1]
           
           f.write(f"CLUSTERING ANALYSIS\n")
           f.write("-"*40 + "\n")
           f.write(f"Total Clusters: {len(clusters)}\n")
           f.write(f"Multi-Factor Clusters: {len(multi_clusters)}\n")
           
           if multi_clusters:
               f.write("\nMulti-Factor Clusters:\n")
               for cluster in multi_clusters:
                   f.write(f"\nCluster {cluster['cluster_id']} (Size: {cluster['size']}):\n")
                   f.write(f"  Selected: {cluster['selected']}\n")
                   f.write(f"  All Factors: {', '.join(cluster['factors'])}\n")
                   if 'avg_correlation' in cluster:
                       f.write(f"  Avg Correlation: {cluster['avg_correlation']:.3f}\n")
           f.write("\n")
       
       # VIF Analysis
       if 'vif_analysis' in results['processing_details']:
           vif_results = results['processing_details']['vif_analysis']
           removed_factors = [r for r in vif_results if r['action'] == 'removed']
           
           f.write(f"VARIANCE INFLATION FACTOR ANALYSIS\n")
           f.write("-"*40 + "\n")
           f.write(f"Factors Removed: {len(removed_factors)}\n")
           
           if removed_factors:
               f.write("\nRemoved Factors (High Multicollinearity):\n")
               for i, result in enumerate(removed_factors, 1):
                   f.write(f"  {i:2d}. {result['factor']}: VIF = {result['vif']:.2f}\n")
           f.write("\n")
       
       # Category Balancing
       if 'category_balancing' in results['processing_details']:
           balancing_info = results['processing_details']['category_balancing']
           f.write(f"CATEGORY BALANCING\n")
           f.write("-"*40 + "\n")
           
           for category, info in balancing_info.items():
               if info['original'] != info['final']:
                   f.write(f"{category}: {info['original']} -> {info['final']} factors\n")
               else:
                   f.write(f"{category}: {info['final']} factors (no change)\n")
           f.write("\n")
       
       # Data Matrices Information
       f.write("DATA MATRICES INFORMATION\n")
       f.write("-"*40 + "\n")
       f.write(f"Original DataFrames: Available in results['original_dataframes']\n")
       f.write(f"Corrected DataFrames: Available in results['corrected_dataframes']\n")
       f.write(f"Factor Matrix: Available in results['factor_matrix']\n")
       f.write(f"  - Shape: {results['factor_matrix'].shape}\n")
       f.write(f"  - Includes: Directionality corrections + Cross-sectional standardization\n")
       f.write(f"  - Ready for: ML models, risk analysis, portfolio optimization\n\n")
       
       f.write(f"Correlation Matrix: Available in results['correlation_matrix']\n")
       f.write(f"  - Shape: {results['correlation_matrix'].shape}\n")
       f.write(f"  - Method: Spearman correlation (robust to outliers)\n")
       f.write(f"  - Based on: Directionally corrected and standardized data\n\n")
       
       f.write("="*80 + "\n")
       f.write("END OF REPORT\n")
       f.write("="*80 + "\n")
   
   print(f"✓ Comprehensive report saved to: {output_file}")


def export_selected_factors_data(dataFrames: Dict[str, pd.DataFrame], 
                               selected_factors: List[str],
                               output_folder: str = 'SelectedFactors') -> None:
    """
    Export only the selected factors to Excel files for further analysis.
    
    Parameters:
    -----------
    dataFrames : Dict[str, pd.DataFrame]
        Original data dictionary
    selected_factors : List[str]
        Selected factors to export
    output_folder : str
        Output folder path
    """
    
    import os
    
    print(f"Exporting {len(selected_factors)} selected factors...")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Export each selected factor
    exported_count = 0
    for factor in selected_factors:
        if factor in dataFrames:
            try:
                output_path = os.path.join(output_folder, f"{factor}.xlsx")
                dataFrames[factor].to_excel(output_path, index=True)
                exported_count += 1
                print(f"  ✓ Exported {factor}")
            except Exception as e:
                print(f"  ✗ Failed to export {factor}: {e}")
    
    # Create summary file
    try:
        summary_path = os.path.join(output_folder, "selected_factors_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("SELECTED FACTORS SUMMARY\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total Selected Factors: {len(selected_factors)}\n")
            f.write(f"Successfully Exported: {exported_count}\n\n")
            f.write("Selected Factors:\n")
            f.write("-"*20 + "\n")
            for i, factor in enumerate(sorted(selected_factors), 1):
                f.write(f"{i:2d}. {factor}\n")
        
        print(f"  ✓ Summary saved to {summary_path}")
    except Exception as e:
        print(f"  ⚠️  Could not create summary file: {e}")
    
    print(f"\n✓ Export completed: {exported_count}/{len(selected_factors)} factors")
    print(f"✓ Files saved to: {output_folder}/")


# Advanced analysis functions for post-selection validation
def validate_factor_selection(factor_matrix: pd.DataFrame, 
                            selected_factors: List[str],
                            correlation_threshold: float = 0.85) -> Dict[str, Any]:
    """
    Validate the quality of factor selection by checking final correlations
    and other statistical properties.
    
    Parameters:
    -----------
    factor_matrix : pd.DataFrame
        Complete factor matrix
    selected_factors : List[str]
        Selected factors to validate
    correlation_threshold : float
        Correlation threshold for validation
        
    Returns:
    --------
    Dict[str, Any] : Validation results
    """
    
    print("Validating factor selection quality...")
    
    # Get selected factor data
    selected_data = factor_matrix[selected_factors].dropna()
    
    # Calculate final correlations
    final_corr = selected_data.corr(method='spearman')
    
    # Check for remaining high correlations
    high_corr_pairs = []
    n_factors = len(selected_factors)
    
    for i in range(n_factors):
        for j in range(i+1, n_factors):
            corr_val = final_corr.iloc[i, j]
            if abs(corr_val) >= correlation_threshold:
                high_corr_pairs.append({
                    'factor1': selected_factors[i],
                    'factor2': selected_factors[j],
                    'correlation': corr_val
                })
    
    # Calculate factor statistics
    factor_stats = {}
    for factor in selected_factors:
        data = selected_data[factor]
        factor_stats[factor] = {
            'mean': data.mean(),
            'std': data.std(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'missing_pct': data.isna().mean()
        }
    
    # Overall validation metrics
    validation_results = {
        'n_factors': len(selected_factors),
        'n_observations': len(selected_data),
        'high_correlations_remaining': len(high_corr_pairs),
        'max_abs_correlation': final_corr.abs().values[np.triu_indices_from(final_corr.values, k=1)].max(),
        'avg_abs_correlation': final_corr.abs().values[np.triu_indices_from(final_corr.values, k=1)].mean(),
        'correlation_matrix': final_corr,
        'high_corr_pairs': high_corr_pairs,
        'factor_statistics': factor_stats
    }
    
    # Print validation summary
    print(f"✓ Validation completed:")
    print(f"  - Final factor count: {validation_results['n_factors']}")
    print(f"  - Observations: {validation_results['n_observations']:,}")
    print(f"  - Remaining high correlations: {validation_results['high_correlations_remaining']}")
    print(f"  - Max absolute correlation: {validation_results['max_abs_correlation']:.3f}")
    print(f"  - Average absolute correlation: {validation_results['avg_abs_correlation']:.3f}")
    
    return validation_results


#%%
# Initialize framework
framework = CountryFactorSelectionFramework(
    correlation_threshold=0.85,
    vif_threshold=5.0,
    min_data_coverage=0.6  # 60% minimum data coverage

)

#%%
# Run complete analysis on your DataFrames
results = framework.run_complete_analysis(transformed_dataFrames)


#%%
# Get directionality report
directionality_report = framework.get_directionality_report()

# Get Factors pre balance catgories
vif_factors=results['vif_filtered_factors']

# Get selected factors
selected_factors = results['final_factors']

#%% Exporting

# Export selected factors
export_selected_factors_data(transformed_dataFrames, selected_factors, 'SelectedFactors')

# Export selected factors - directionality corrected
export_selected_factors_data(directionality_corrected_dataFrames, selected_factors, 'SelectedFactorsDirectionalityCorrected')

# Export selected factors
export_selected_factors_data(transformed_dataFrames, vif_factors, 'SelectedFactorsVIF')

# Export selected factors - directionality corrected
export_selected_factors_data(directionality_corrected_dataFrames, vif_factors, 'SelectedFactorsDirectionalityCorrectedVIF')


#%%
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings

class FactorBalancing:
    """
    Advanced factor balancing framework for quantitative finance.
    
    Provides multiple sophisticated methods to balance factors across categories
    while maintaining statistical rigor and portfolio diversification principles.
    """
    
    def __init__(self, 
                 min_factors_per_category: int = 1,
                 max_factors_per_category: Optional[int] = None,
                 default_importance_weights: Optional[Dict[str, float]] = None,
                 random_state: Optional[int] = 42):
        """
        Initialize the FactorBalancing framework.
        
        Parameters:
        -----------
        min_factors_per_category : int, default 1
            Minimum factors required per category
        max_factors_per_category : int, optional
            Maximum factors allowed per category (calculated adaptively if None)
        default_importance_weights : Dict[str, float], optional
            Default category importance weights
        random_state : int, optional
            Random seed for reproducibility
        """
        self.min_factors_per_category = min_factors_per_category
        self.max_factors_per_category = max_factors_per_category
        self.random_state = random_state
        
        # Set default importance weights based on financial theory
        self.default_importance_weights = default_importance_weights or {
            'Valuation': 3.0,      # Core for value investing
            'Quality': 2.5,        # Important for risk management
            'Profitability': 2.5,  # Key for fundamentals
            'Momentum': 2.0,       # Important for timing
            'Risk': 2.0,          # Critical for risk management
            'Size': 1.5,          # Less critical but useful
            'Sentiment': 1.0,     # Supplementary
            'Macro': 1.5,         # Context-dependent
            'Unknown': 0.5        # Lowest priority
        }
        
        # Results storage
        self.results = {}
        self.balancing_history = []
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def balance_factors(self, 
                       selected_factors: List[str], 
                       classification_map: Dict[str, str],
                       factor_matrix: pd.DataFrame,
                       method: str = 'information_based',
                       target_factors_per_category: Optional[Dict[str, int]] = None,
                       total_target_factors: Optional[int] = None,
                       importance_weights: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Main method to balance factors across categories using specified method.
        
        Parameters:
        -----------
        selected_factors : List[str]
            Previously selected factors to balance
        classification_map : Dict[str, str]
            Mapping from factor names to categories
        factor_matrix : pd.DataFrame
            Standardized factor matrix for quality assessment
        method : str, default 'information_based'
            Balancing method: 'information_based', 'correlation_based', 
            'risk_budgeting', 'equal_weight', or 'adaptive'
        target_factors_per_category : Dict[str, int], optional
            Specific targets per category (overrides adaptive calculation)
        total_target_factors : int, optional
            Total target number of factors
        importance_weights : Dict[str, float], optional
            Category importance weights (overrides defaults)
            
        Returns:
        --------
        List[str] : Category-balanced factors
        """
        print(f"Starting factor balancing with {method} method...")
        
        # Group factors by category
        category_groups = self._group_factors_by_category(selected_factors, classification_map)
        
        # Calculate adaptive limits and targets
        if self.max_factors_per_category is None:
            total_factors = len(selected_factors)
            n_categories = len(category_groups)
            self.max_factors_per_category = max(2, total_factors // n_categories + 2)
        
        # Use provided targets or calculate adaptive ones
        if target_factors_per_category is None and method == 'adaptive':
            target_factors_per_category = self.calculate_adaptive_targets(
                category_groups, factor_matrix, total_target_factors, importance_weights
            )
        elif target_factors_per_category is None:
            target_factors_per_category = {}
        
        # Apply selected balancing method
        if method == 'information_based':
            balanced_factors = self._information_based_balancing(
                category_groups, factor_matrix, target_factors_per_category
            )
        elif method == 'correlation_based':
            balanced_factors = self._correlation_based_balancing(
                category_groups, factor_matrix, target_factors_per_category
            )
        elif method == 'risk_budgeting':
            balanced_factors = self._risk_budgeting_balancing(
                category_groups, factor_matrix, target_factors_per_category
            )
        elif method == 'equal_weight':
            balanced_factors = self._equal_weight_balancing(
                category_groups, target_factors_per_category
            )
        elif method == 'adaptive':
            # Adaptive uses information_based with adaptive targets
            balanced_factors = self._information_based_balancing(
                category_groups, factor_matrix, target_factors_per_category
            )
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Store results and history
        self._store_balancing_results(selected_factors, balanced_factors, category_groups, 
                                    method, target_factors_per_category)
        
        print(f"Factor balancing completed: {len(selected_factors)} → {len(balanced_factors)} factors")
        return balanced_factors
    
    def _group_factors_by_category(self, factors: List[str], 
                                 classification_map: Dict[str, str]) -> Dict[str, List[str]]:
        """Group factors by their categories."""
        category_groups = {}
        for factor in factors:
            category = classification_map.get(factor, 'Unknown')
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(factor)
        return category_groups
    
    def calculate_adaptive_targets(self, 
                                 category_groups: Dict[str, List[str]], 
                                 factor_matrix: pd.DataFrame,
                                 total_target_factors: Optional[int] = None,
                                 importance_weights: Optional[Dict[str, float]] = None) -> Dict[str, int]:
        """
        Calculate adaptive targets per category based on data quality, 
        category importance, and availability.
        """
        if total_target_factors is None:
            total_target_factors = sum(len(factors) for factors in category_groups.values()) // 2
        
        # Use provided weights or defaults
        weights = importance_weights or self.default_importance_weights
        
        # Calculate scores for each category
        category_scores = {}
        
        for category, factors in category_groups.items():
            # Base importance
            importance = weights.get(category, 1.0)
            
            # Data quality score
            if len(factors) > 0 and not factor_matrix.empty:
                available_factors = [f for f in factors if f in factor_matrix.columns]
                if available_factors:
                    data_quality = factor_matrix[available_factors].notna().mean().mean()
                    data_coverage = len(available_factors) / len(factors)
                else:
                    data_quality = 0.0
                    data_coverage = 0.0
            else:
                data_quality = 0.5
                data_coverage = 1.0
            
            # Availability score (normalize by having at least 5 factors as ideal)
            availability = min(1.0, len(factors) / 5)
            
            # Combined score
            category_scores[category] = importance * data_quality * data_coverage * availability
        
        # Allocate targets proportionally
        total_score = sum(category_scores.values())
        targets = {}
        
        for category, score in category_scores.items():
            if total_score > 0:
                raw_target = (score / total_score) * total_target_factors
                # Ensure at least min factors per category, cap at available factors
                targets[category] = max(self.min_factors_per_category, 
                                      min(len(category_groups[category]), 
                                          round(raw_target)))
            else:
                targets[category] = self.min_factors_per_category
        
        # Adjust if total exceeds target
        current_total = sum(targets.values())
        if current_total > total_target_factors:
            # Proportionally reduce, maintaining minimum per category
            excess = current_total - total_target_factors
            reducible_total = current_total - len(targets) * self.min_factors_per_category
            
            if reducible_total > 0:
                reduction_factor = max(0, 1 - excess / reducible_total)
                for category in targets:
                    if targets[category] > self.min_factors_per_category:
                        reducible = targets[category] - self.min_factors_per_category
                        targets[category] = self.min_factors_per_category + round(reducible * reduction_factor)
        
        print("Adaptive targets calculated:")
        for category, target in targets.items():
            available = len(category_groups[category])
            score = category_scores[category]
            print(f"  {category:<15}: {target:2d}/{available:2d} factors (score: {score:.2f})")
        
        return targets
    
    def _information_based_balancing(self, 
                                   category_groups: Dict[str, List[str]], 
                                   factor_matrix: pd.DataFrame,
                                   target_per_category: Dict[str, int]) -> List[str]:
        """
        Balance based on information content using PCA and mutual information.
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("Warning: scikit-learn not available, falling back to variance-based selection")
            return self._variance_based_fallback(category_groups, factor_matrix, target_per_category)
        
        balanced_factors = []
        
        for category, factors in category_groups.items():
            target_count = self._get_target_count(category, factors, target_per_category)
            
            if len(factors) <= target_count:
                balanced_factors.extend(factors)
                continue
            
            # Get factor data
            factor_data = factor_matrix[factors].dropna()
            
            if len(factor_data) == 0:
                balanced_factors.extend(factors[:target_count])
                continue
            
            # Calculate factor importance using PCA
            try:
                n_components = min(len(factors), len(factor_data), 3)
                pca = PCA(n_components=n_components, random_state=self.random_state)
                pca.fit(factor_data)
                
                # Calculate each factor's contribution to principal components
                loadings = pca.components_
                factor_importance = np.sum(np.abs(loadings), axis=0)
                
            except Exception as e:
                print(f"    PCA failed for {category}, using variance: {e}")
                factor_importance = factor_data.var().values
            
            # Iterative selection to maximize information while minimizing correlation
            selected_factors_this_category = []
            remaining_factors = factors.copy()
            
            for i in range(target_count):
                if not remaining_factors:
                    break
                    
                if i == 0:
                    # First factor: highest information content
                    factor_scores = {f: factor_importance[factors.index(f)] 
                                   for f in remaining_factors}
                    best_factor = max(factor_scores, key=factor_scores.get)
                else:
                    # Subsequent factors: balance information content with diversity
                    scores = {}
                    selected_data = factor_data[selected_factors_this_category]
                    
                    for factor in remaining_factors:
                        # Information score (normalized)
                        info_score = factor_importance[factors.index(factor)]
                        max_info = max(factor_importance)
                        norm_info_score = info_score / max_info if max_info > 0 else 0
                        
                        # Correlation penalty
                        if len(selected_factors_this_category) > 0:
                            correlations = np.abs(factor_data[factor].corr(selected_data, method='spearman'))
                            max_corr = correlations.max() if len(correlations) > 0 else 0
                        else:
                            max_corr = 0
                        
                        # Combined score (information content - correlation penalty)
                        scores[factor] = norm_info_score * (1 - max_corr)
                    
                    best_factor = max(scores, key=scores.get) if scores else remaining_factors[0]
                
                selected_factors_this_category.append(best_factor)
                remaining_factors.remove(best_factor)
            
            balanced_factors.extend(selected_factors_this_category)
            print(f"  {category:<15}: {len(factors):2d} → {len(selected_factors_this_category):2d} factors (information-based)")
        
        return balanced_factors
    
    def _correlation_based_balancing(self, 
                                   category_groups: Dict[str, List[str]], 
                                   factor_matrix: pd.DataFrame,
                                   target_per_category: Dict[str, int]) -> List[str]:
        """
        Balance based on correlation structure using hierarchical clustering.
        """
        balanced_factors = []
        
        for category, factors in category_groups.items():
            target_count = self._get_target_count(category, factors, target_per_category)
            
            if len(factors) <= target_count:
                balanced_factors.extend(factors)
                continue
            
            # Get correlation matrix for this category
            factor_data = factor_matrix[factors].dropna()
            if len(factor_data) == 0:
                balanced_factors.extend(factors[:target_count])
                continue
            
            corr_matrix = factor_data.corr(method='spearman').fillna(0)
            
            try:
                # Convert correlation to distance
                distance_matrix = 1 - np.abs(corr_matrix)
                np.fill_diagonal(distance_matrix.values, 0)
                
                # Hierarchical clustering
                condensed_distances = squareform(distance_matrix.values)
                linkage_matrix = linkage(condensed_distances, method='ward')
                
                # Cut tree to get desired number of clusters
                cluster_labels = fcluster(linkage_matrix, target_count, criterion='maxclust')
                
                # Select one representative from each cluster
                selected_factors_this_category = []
                for cluster_id in range(1, target_count + 1):
                    cluster_factors = [factors[i] for i, label in enumerate(cluster_labels) 
                                     if label == cluster_id]
                    
                    if cluster_factors:
                        # Select factor with highest variance (most informative)
                        if len(cluster_factors) == 1:
                            best_factor = cluster_factors[0]
                        else:
                            factor_vars = {f: factor_data[f].var() for f in cluster_factors}
                            best_factor = max(factor_vars, key=factor_vars.get)
                        selected_factors_this_category.append(best_factor)
                
            except Exception as e:
                print(f"    Clustering failed for {category}, using correlation ranking: {e}")
                # Fallback: select factors with lowest average correlation
                avg_abs_corr = corr_matrix.abs().mean(axis=1)
                selected_factors_this_category = avg_abs_corr.nsmallest(target_count).index.tolist()
            
            balanced_factors.extend(selected_factors_this_category)
            print(f"  {category:<15}: {len(factors):2d} → {len(selected_factors_this_category):2d} factors (correlation-based)")
        
        return balanced_factors
    
    def _risk_budgeting_balancing(self, 
                                category_groups: Dict[str, List[str]], 
                                factor_matrix: pd.DataFrame,
                                target_per_category: Dict[str, int]) -> List[str]:
        """
        Balance based on risk contribution to prevent category domination.
        """
        balanced_factors = []
        
        # Calculate overall risk metrics
        all_factors = [f for factors in category_groups.values() for f in factors]
        factor_data = factor_matrix[all_factors].dropna()
        
        if len(factor_data) == 0:
            return all_factors
        
        # Calculate covariance matrix
        try:
            cov_matrix = factor_data.cov()
        except Exception:
            # Fallback to correlation-based approach
            return self._correlation_based_balancing(category_groups, factor_matrix, target_per_category)
        
        for category, factors in category_groups.items():
            target_count = target_per_category.get(category)
            
            if target_count is None:
                # Risk budgeting approach: allocate based on category's risk contribution
                category_factors_in_data = [f for f in factors if f in factor_data.columns]
                if category_factors_in_data:
                    category_data = factor_data[category_factors_in_data]
                    category_risk = np.sqrt(np.diag(category_data.cov())).mean()
                    total_risk = np.sqrt(np.diag(cov_matrix)).mean()
                    
                    # Risk-based allocation with caps
                    risk_weight = min(0.3, category_risk / total_risk if total_risk > 0 else 0.1)
                    target_count = max(self.min_factors_per_category, 
                                     min(self.max_factors_per_category, 
                                         int(len(factors) * risk_weight + 0.5)))
                else:
                    target_count = self.min_factors_per_category
            
            if len(factors) <= target_count:
                balanced_factors.extend(factors)
                continue
            
            # Calculate risk contribution for each factor
            factor_risks = {}
            for factor in factors:
                if factor in cov_matrix.index:
                    # Risk contribution = factor's systematic risk
                    other_factors = [f for f in all_factors if f != factor and f in cov_matrix.index]
                    if other_factors:
                        cross_cov = cov_matrix.loc[factor, other_factors].abs().mean()
                        own_var = cov_matrix.loc[factor, factor]
                        risk_contrib = np.sqrt(own_var + cross_cov)
                    else:
                        risk_contrib = np.sqrt(cov_matrix.loc[factor, factor])
                else:
                    # Fallback to standard deviation
                    if factor in factor_data.columns:
                        risk_contrib = factor_data[factor].std()
                    else:
                        risk_contrib = 1.0
                
                factor_risks[factor] = risk_contrib
            
            # Select factors with highest risk contribution (most systematic)
            selected_factors_this_category = sorted(factor_risks, 
                                                  key=factor_risks.get, 
                                                  reverse=True)[:target_count]
            
            balanced_factors.extend(selected_factors_this_category)
            print(f"  {category:<15}: {len(factors):2d} → {len(selected_factors_this_category):2d} factors (risk-budgeting)")
        
        return balanced_factors
    
    def _equal_weight_balancing(self, 
                              category_groups: Dict[str, List[str]], 
                              target_per_category: Dict[str, int]) -> List[str]:
        """
        Simple equal-weight balancing across categories.
        """
        total_factors = sum(len(factors) for factors in category_groups.values())
        n_categories = len(category_groups)
        
        # Calculate base allocation
        base_allocation = max(self.min_factors_per_category, total_factors // n_categories)
        remainder = total_factors % n_categories
        
        balanced_factors = []
        categories_sorted = sorted(category_groups.keys())
        
        for i, category in enumerate(categories_sorted):
            factors = category_groups[category]
            
            if category in target_per_category:
                target_count = target_per_category[category]
            else:
                # Equal allocation with remainder distributed to first categories
                target_count = base_allocation + (1 if i < remainder else 0)
                target_count = max(self.min_factors_per_category, 
                                 min(self.max_factors_per_category or len(factors), 
                                     target_count))
            
            # Simple selection (could be enhanced with factor quality metrics)
            selected_factors_this_category = factors[:min(target_count, len(factors))]
            balanced_factors.extend(selected_factors_this_category)
            
            print(f"  {category:<15}: {len(factors):2d} → {len(selected_factors_this_category):2d} factors (equal-weight)")
        
        return balanced_factors
    
    def _variance_based_fallback(self, 
                               category_groups: Dict[str, List[str]], 
                               factor_matrix: pd.DataFrame,
                               target_per_category: Dict[str, int]) -> List[str]:
        """
        Fallback method using variance when advanced methods fail.
        """
        balanced_factors = []
        
        for category, factors in category_groups.items():
            target_count = self._get_target_count(category, factors, target_per_category)
            
            if len(factors) <= target_count:
                balanced_factors.extend(factors)
                continue
            
            # Select factors with highest variance
            available_factors = [f for f in factors if f in factor_matrix.columns]
            if available_factors:
                factor_data = factor_matrix[available_factors].dropna()
                if len(factor_data) > 0:
                    factor_vars = factor_data.var().sort_values(ascending=False)
                    selected_factors_this_category = factor_vars.head(target_count).index.tolist()
                else:
                    selected_factors_this_category = available_factors[:target_count]
            else:
                selected_factors_this_category = factors[:target_count]
            
            balanced_factors.extend(selected_factors_this_category)
            print(f"  {category:<15}: {len(factors):2d} → {len(selected_factors_this_category):2d} factors (variance-based)")
        
        return balanced_factors
    
    def _get_target_count(self, category: str, factors: List[str], 
                         target_per_category: Dict[str, int]) -> int:
        """Get target count for a category."""
        if category in target_per_category:
            return target_per_category[category]
        else:
            return max(self.min_factors_per_category, 
                      min(self.max_factors_per_category or len(factors), 
                          len(factors) // 2))
    
    def _store_balancing_results(self, original_factors: List[str], balanced_factors: List[str],
                               category_groups: Dict[str, List[str]], method: str,
                               targets: Dict[str, int]) -> None:
        """Store detailed results of the balancing process."""
        
        # Calculate detailed statistics
        balancing_details = {}
        for category, factors in category_groups.items():
            selected_from_category = [f for f in balanced_factors if f in factors]
            balancing_details[category] = {
                'original_count': len(factors),
                'selected_count': len(selected_from_category),
                'target_count': targets.get(category, 'adaptive'),
                'selected_factors': selected_from_category,
                'rejected_factors': [f for f in factors if f not in selected_from_category],
                'selection_rate': len(selected_from_category) / len(factors) if factors else 0
            }
        
        # Store current results
        self.results = {
            'method': method,
            'original_factor_count': len(original_factors),
            'balanced_factor_count': len(balanced_factors),
            'reduction_ratio': len(balanced_factors) / len(original_factors),
            'category_details': balancing_details,
            'balanced_factors': balanced_factors,
            'targets_used': targets
        }
        
        # Add to history
        self.balancing_history.append({
            'method': method,
            'timestamp': pd.Timestamp.now(),
            'input_factors': len(original_factors),
            'output_factors': len(balanced_factors),
            'categories_processed': len(category_groups)
        })
    
    def get_balancing_summary(self) -> pd.DataFrame:
        """
        Get a summary of the last balancing operation.
        
        Returns:
        --------
        pd.DataFrame : Summary of balancing results by category
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for category, details in self.results['category_details'].items():
            summary_data.append({
                'Category': category,
                'Original_Count': details['original_count'],
                'Selected_Count': details['selected_count'],
                'Target_Count': details['target_count'],
                'Selection_Rate': f"{details['selection_rate']:.1%}",
                'Selected_Factors': ', '.join(details['selected_factors'][:3]) + 
                                  ('...' if len(details['selected_factors']) > 3 else '')
            })
        
        return pd.DataFrame(summary_data)
    
    def get_history(self) -> pd.DataFrame:
        """
        Get history of all balancing operations.
        
        Returns:
        --------
        pd.DataFrame : History of balancing operations
        """
        return pd.DataFrame(self.balancing_history)
    
    def compare_methods(self, 
                       selected_factors: List[str], 
                       classification_map: Dict[str, str],
                       factor_matrix: pd.DataFrame,
                       methods: List[str] = None) -> Dict[str, List[str]]:
        """
        Compare different balancing methods on the same input.
        
        Parameters:
        -----------
        selected_factors : List[str]
            Input factors to balance
        classification_map : Dict[str, str]
            Factor classification mapping
        factor_matrix : pd.DataFrame
            Factor data matrix
        methods : List[str], optional
            Methods to compare (default: all methods)
            
        Returns:
        --------
        Dict[str, List[str]] : Results from each method
        """
        if methods is None:
            methods = ['information_based', 'correlation_based', 'risk_budgeting', 
                      'equal_weight', 'adaptive']
        
        print("Comparing balancing methods...")
        print("=" * 50)
        
        comparison_results = {}
        
        for method in methods:
            print(f"\nTesting {method} method:")
            try:
                result = self.balance_factors(
                    selected_factors=selected_factors,
                    classification_map=classification_map,
                    factor_matrix=factor_matrix,
                    method=method
                )
                comparison_results[method] = result
                print(f"  Result: {len(selected_factors)} → {len(result)} factors")
                
            except Exception as e:
                print(f"  Failed: {e}")
                comparison_results[method] = []
        
        print("\nComparison Summary:")
        print("-" * 30)
        for method, result in comparison_results.items():
            print(f"{method:<20}: {len(result):3d} factors")
        
        return comparison_results




#%%
fb= FactorBalancing()

#information_based_factors = fb.balance_factors(vif_factors, classification_map, final_matrix_unbalanced, method='information_based')

correlation_based_factors = fb.balance_factors(vif_factors, classification_map, final_matrix_unbalanced, method='correlation_based')
#comparison_results= fb.compare_methods(vif_factors, classification_map, final_matrix_unbalanced)


#%%
# Dataframes with signal direction applied
#directionality_corrected_dataFrames =  results['corrected_dataframes']
# Create final modeling matrix ---> category balancing
#final_matrix_balanced = framework.create_final_factor_matrix(directionality_corrected_dataFrames, selected_factors)
# Create final modeling matrix ---> NO category balancing
#final_matrix_unbalanced = framework.create_final_factor_matrix(directionality_corrected_dataFrames, vif_factors)
#final_matrix_balanced.to_excel('final_matrix_balanced.xlsx')
#final_matrix_unbalanced.to_excel('final_matrix_unbalanced.xlsx')
#final_matrix_unbalanced=pd.read_excel('final_matrix_unbalanced.xlsx', header=0, index_col=[0,1])

#%%
# Plot final correlation matrix
framework.plot_correlation_heatmap(results['correlation_matrix'], selected_factors)

#%%


#%%
# Validate selection quality
validation = validate_factor_selection(
    results['factor_matrix'], 
    selected_factors)

# Generate comprehensive report
create_factor_analysis_report(results, 'factor_analysis_report.txt')

classification_map= results['classification_map']


#%%
































#%%

# Initialize and run category generator
category_generator = fm.CategorySignalGenerator(
    data=final_matrix,
    classification_map=classification_map
)

# Run the complete category signals
category_signals = category_generator.generate_category_signals()

# Get statistics
stats_df = category_generator.get_signal_statistics()
print(stats_df)


#%% #BACKTEST SIGNAL




#%%
#Create final alpha signal
alpha_generator = fm.AlphaSignalGenerator(category_signals)
final_alpha = alpha_generator.create_final_alpha_signal(method='equal_weighted')

#%%
#: Analyze and visualize
analyzer = fm.SignalAnalyzer(category_signals, final_alpha)
contributions_analysis = analyzer.calculate_category_contributions()

# View contribution statistics
print("Contribution Statistics:")
print(contributions_analysis['contribution_stats'])

print("\nRelative Importance:")
print(contributions_analysis['relative_importance'])

# Plot comprehensive analysis
analyzer.plot_category_contributions()

# Plot for specific period/countries
analyzer.plot_category_contributions(
    date_range=('2020-01-01', '2024-01-01'),
    countries=['Japan']
)

# Plot distributions (optional, requires matplotlib)
analyzer.plot_signal_distributions()

