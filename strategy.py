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


#%%

# Execute the main pipeline
processed_data, regions_dict, classification = run_inputs()

# Make key variables available in the global scope for interactive use
dataFrames = processed_data

print(f"\n📋 VARIABLES AVAILABLE FOR ANALYSIS:")
print(f"   • dataFrames: {len(dataFrames)} processed datasets")
print(f"   • regions_dict: Regional country groupings")
print(f"   • classification: Country classification DataFrame")
print(f"   • processed_data: Same as dataFrames (alias)")

#%%
#Quick data exploration
fm.explore_data(dataFrames, regions_dict)

#%%

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import warnings


class QuantMetricsTransformer:
    """
    A comprehensive class for transforming quantitative finance metrics.
    
    This class handles the transformation of financial metrics based on predefined rules:
    - Absolute: No transformation (ratios, yields, etc.)
    - Percent: Point-to-point percentage changes over 3M and 12M periods
    - Difference: Point-to-point differences over 3M and 12M periods
    
    Attributes:
        THREE_MONTH_DAYS (int): Business days in 3 months (63)
        TWELVE_MONTH_DAYS (int): Business days in 12 months (252)
    """
    
    # Class constants
    THREE_MONTH_DAYS = 63
    TWELVE_MONTH_DAYS = 252
    
    def __init__(self, 
                 dataframes: Optional[Dict[str, pd.DataFrame]] = None,
                 classification_map: Optional[Dict[str, str]] = None,
                 transformation_map: Optional[Dict[str, str]] = None):
        """
        Initialize the QuantMetricsTransformer.
        
        Parameters:
        -----------
        dataframes : dict, optional
            Dictionary with metric names as keys and DataFrames as values
        classification_map : dict, optional
            Dictionary mapping metrics to factor categories
        transformation_map : dict, optional
            Dictionary specifying transformation type for each metric
        """
        self.original_dataframes = dataframes or {}
        self.transformed_dataframes = {}
        self.new_classification_map = {}
        
        # Set default maps if not provided
        self.classification_map = classification_map or self._get_default_classification_map()
        self.transformation_map = transformation_map or self._get_default_transformation_map()
        
        # Transformation statistics
        self.transform_stats = {
            'absolute': 0,
            'percent': 0, 
            'difference': 0,
            'skipped': 0
        }
    
    @staticmethod
    def _get_default_classification_map() -> Dict[str, str]:
        """Return the default classification map for factor categories."""
        return {
            # VALUATION FACTORS
            'PE': 'Valuation', 'Fwd_PE': 'Valuation', 'PB': 'Valuation',
            'PS': 'Valuation', 'Fwd_PS': 'Valuation', 'PCF': 'Valuation',
            'Fwd_PCF': 'Valuation', 'EV_EBIT': 'Valuation', 'EV_EBITDA': 'Valuation',
            'Fwd_EV_EBITDA': 'Valuation', 'EarningsYieldTTM': 'Valuation',
            'EarningsYieldFWD': 'Valuation', 'CashFlowYieldTTM': 'Valuation',
            'CashFlowYieldFWD': 'Valuation', 'DVD': 'Valuation', 'Fwd_DVD': 'Valuation',
            
            # VALUATION SPREADS
            'EarningsYieldTTMSpread': 'Valuation', 'EarningsYieldFWDSpread': 'Valuation',
            'CashFlowYieldTTMSpread': 'Valuation', 'CashFlowYieldFWDSpread': 'Valuation',
            'DvdYieldTTMSpread': 'Valuation', 'DvdYieldFWDSpread': 'Valuation',
            
            # QUALITY FACTORS
            'Debt_to_Equity': 'Quality', 'Net_Debt_Ebitda': 'Quality',
            'AssetsEquity': 'Quality', 'Debt': 'Quality', 'Equity': 'Quality',
            'Liabilities': 'Quality', 'CF': 'Quality', 'FwdCF': 'Quality',
            
            # PROFITABILITY FACTORS
            'EbitMargin': 'Profitability', 'EbitdaMargin': 'Profitability',
            'NetMargin': 'Profitability', 'FwdEBITDAMargin': 'Profitability',
            'FwdNetMargin': 'Profitability', 'EBIT': 'Profitability',
            'EBITDA': 'Profitability', 'FwdEBITDA': 'Profitability',
            'EPS': 'Profitability', 'Earnings': 'Profitability',
            'FwdEarnings': 'Profitability', 'ROE': 'Profitability',
            'Fwd_ROE': 'Profitability', 'Return_Capital': 'Profitability',
            
            # MOMENTUM FACTORS
            'ConsensusSalesGrowth': 'Momentum', 'ConsensusEbitdaGrowth': 'Momentum',
            'ConsensusEarningsGrowth': 'Momentum', 'ConsensusCashFlowGrowth': 'Momentum',
            'RollingEarnings': 'Momentum', 'FwdRollingEarnings': 'Momentum',
            'CumFlow': 'Momentum', 'Flows': 'Momentum',
            
            # SIZE FACTORS
            'Market_Cap': 'Size', 'EV': 'Size', 'Revenue': 'Size',
            'FwdRevenue': 'Size', 'Price': 'Size', 'Assets': 'Size',
            
            # RISK FACTORS
            'RollingVol': 'Risk', 'Ten_Year': 'Risk',
            
            # SENTIMENT FACTORS
            'SI': 'Sentiment', 'SI_Ratio': 'Sentiment',
            
            # MACRO FACTORS
            'GDP': 'Macro', 'M2': 'Macro'
        }
    
    @staticmethod
    def _get_default_transformation_map() -> Dict[str, str]:
        """Return the default transformation map."""
        return {
            # ABSOLUTE - No transformation needed
            'PE': 'absolute', 'Fwd_PE': 'absolute', 'PB': 'absolute',
            'PS': 'absolute', 'Fwd_PS': 'absolute', 'PCF': 'absolute',
            'Fwd_PCF': 'absolute', 'EV_EBIT': 'absolute', 'EV_EBITDA': 'absolute',
            'Fwd_EV_EBITDA': 'absolute', 'EarningsYieldTTM': 'absolute',
            'EarningsYieldFWD': 'absolute', 'CashFlowYieldTTM': 'absolute',
            'CashFlowYieldFWD': 'absolute', 'DVD': 'absolute', 'Fwd_DVD': 'absolute',
            'Debt_to_Equity': 'absolute', 'Net_Debt_Ebitda': 'absolute',
            'AssetsEquity': 'absolute', 'SI': 'absolute', 'SI_Ratio': 'absolute',
            
            # DIFFERENCE - Point-to-point differences
            'EarningsYieldTTMSpread': 'difference', 'EarningsYieldFWDSpread': 'difference',
            'CashFlowYieldTTMSpread': 'difference', 'CashFlowYieldFWDSpread': 'difference',
            'DvdYieldTTMSpread': 'difference', 'DvdYieldFWDSpread': 'difference',
            'EbitMargin': 'difference', 'EbitdaMargin': 'difference',
            'NetMargin': 'difference', 'FwdEBITDAMargin': 'difference',
            'FwdNetMargin': 'difference', 'ROE': 'difference', 'Fwd_ROE': 'difference',
            'Return_Capital': 'difference', 'ConsensusSalesGrowth': 'difference',
            'ConsensusEbitdaGrowth': 'difference', 'ConsensusEarningsGrowth': 'difference',
            'ConsensusCashFlowGrowth': 'difference', 'RollingEarnings': 'difference',
            'FwdRollingEarnings': 'difference', 'RollingVol': 'difference',
            'Ten_Year': 'difference', 'CumFlow': 'difference', 'Flows': 'difference',
            
            # PERCENT - Percentage changes
            'Debt': 'percent', 'Equity': 'percent', 'Liabilities': 'percent',
            'CF': 'percent', 'FwdCF': 'percent', 'EBIT': 'percent',
            'EBITDA': 'percent', 'FwdEBITDA': 'percent', 'EPS': 'percent',
            'Earnings': 'percent', 'FwdEarnings': 'percent', 'Market_Cap': 'percent',
            'EV': 'percent', 'Revenue': 'percent', 'FwdRevenue': 'percent',
            'Price': 'percent', 'Assets': 'percent', 'GDP': 'percent', 'M2': 'percent'
        }
    
    def load_data(self, dataframes: Dict[str, pd.DataFrame]) -> 'QuantMetricsTransformer':
        """
        Load dataframes into the transformer.
        
        Parameters:
        -----------
        dataframes : dict
            Dictionary with metric names as keys and DataFrames as values
            
        Returns:
        --------
        self : QuantMetricsTransformer
            Returns self for method chaining
        """
        self.original_dataframes = dataframes
        return self
    
    def set_classification_map(self, classification_map: Dict[str, str]) -> 'QuantMetricsTransformer':
        """
        Set custom classification map.
        
        Parameters:
        -----------
        classification_map : dict
            Dictionary mapping metrics to factor categories
            
        Returns:
        --------
        self : QuantMetricsTransformer
            Returns self for method chaining
        """
        self.classification_map = classification_map
        return self
    
    def set_transformation_map(self, transformation_map: Dict[str, str]) -> 'QuantMetricsTransformer':
        """
        Set custom transformation map.
        
        Parameters:
        -----------
        transformation_map : dict
            Dictionary specifying transformation type for each metric
            
        Returns:
        --------
        self : QuantMetricsTransformer
            Returns self for method chaining
        """
        self.transformation_map = transformation_map
        return self
    
    def transform(self, slice_months: int = 12) -> 'QuantMetricsTransformer':
        """
        Execute the transformation of all metrics.
        
        Parameters:
        -----------
        slice_months : int, default=12
            Number of months to slice from the end of each dataframe for consistency
            (slicing is applied AFTER transformations to preserve data for calculations)
            
        Returns:
        --------
        self : QuantMetricsTransformer
            Returns self for method chaining
        """
        if not self.original_dataframes:
            raise ValueError("No dataframes loaded. Use load_data() first.")
        
        # Calculate slice period in business days
        slice_days = slice_months * 21  # Approximate business days per month
        
        # Reset transformation results
        self.transformed_dataframes = {}
        self.new_classification_map = {}
        self.transform_stats = {key: 0 for key in self.transform_stats}
        
        # Process each metric (transformation first, then slice)
        for metric_name, df in self.original_dataframes.items():
            self._process_metric(metric_name, df, slice_days)
        
        return self
    
    def _process_metric(self, metric_name: str, df: pd.DataFrame, slice_days: int) -> None:
        """Process a single metric according to its transformation type."""
        
        # Skip if metric not in transformation_map
        if metric_name not in self.transformation_map:
            warnings.warn(f"{metric_name} not found in transformation_map, skipping...")
            self.transform_stats['skipped'] += 1
            return
        
        # Get transformation type and factor category
        transform_type = self.transformation_map[metric_name]
        factor_category = self.classification_map.get(metric_name, 'Unknown')
        
        # Apply transformation FIRST on full dataset, then slice
        if transform_type == 'absolute':
            self._process_absolute(metric_name, df, factor_category, slice_days)
        elif transform_type == 'percent':
            self._process_percent(metric_name, df, factor_category, slice_days)
        elif transform_type == 'difference':
            self._process_difference(metric_name, df, factor_category, slice_days)
        else:
            warnings.warn(f"Unknown transformation type '{transform_type}' for {metric_name}")
            self.transform_stats['skipped'] += 1
    
    def _process_absolute(self, metric_name: str, df: pd.DataFrame, factor_category: str, slice_days: int) -> None:
        """Process metrics with absolute transformation (no change)."""
        # For absolute metrics, just slice the original data
        df_sliced = df.iloc[-slice_days:].copy() if len(df) >= slice_days else df.copy()
        self.transformed_dataframes[metric_name] = df_sliced
        self.new_classification_map[metric_name] = factor_category
        self.transform_stats['absolute'] += 1
    
    def _process_percent(self, metric_name: str, df: pd.DataFrame, factor_category: str, slice_days: int) -> None:
        """Process metrics with percent change transformation."""
        # Apply transformations on full dataset first
        # 3-month percent change
        metric_3mo = f"{metric_name}_3mo_pct_chg"
        df_3mo_full = df.pct_change(periods=self.THREE_MONTH_DAYS) * 100
        # Then slice
        df_3mo = df_3mo_full.iloc[-slice_days:].copy() if len(df_3mo_full) >= slice_days else df_3mo_full.copy()
        self.transformed_dataframes[metric_3mo] = df_3mo
        self.new_classification_map[metric_3mo] = factor_category
        
        # 12-month percent change  
        metric_12mo = f"{metric_name}_12mo_pct_chg"
        df_12mo_full = df.pct_change(periods=self.TWELVE_MONTH_DAYS) * 100
        # Then slice
        df_12mo = df_12mo_full.iloc[-slice_days:].copy() if len(df_12mo_full) >= slice_days else df_12mo_full.copy()
        self.transformed_dataframes[metric_12mo] = df_12mo
        self.new_classification_map[metric_12mo] = factor_category
        
        self.transform_stats['percent'] += 1
    
    def _process_difference(self, metric_name: str, df: pd.DataFrame, factor_category: str, slice_days: int) -> None:
        """Process metrics with difference transformation."""
        # Apply transformations on full dataset first
        # 3-month difference
        metric_3mo = f"{metric_name}_3mo_diff_chg"
        df_3mo_full = df.diff(periods=self.THREE_MONTH_DAYS)
        # Then slice
        df_3mo = df_3mo_full.iloc[-slice_days:].copy() if len(df_3mo_full) >= slice_days else df_3mo_full.copy()
        self.transformed_dataframes[metric_3mo] = df_3mo
        self.new_classification_map[metric_3mo] = factor_category
        
        # 12-month difference
        metric_12mo = f"{metric_name}_12mo_diff_chg"
        df_12mo_full = df.diff(periods=self.TWELVE_MONTH_DAYS)
        # Then slice
        df_12mo = df_12mo_full.iloc[-slice_days:].copy() if len(df_12mo_full) >= slice_days else df_12mo_full.copy()
        self.transformed_dataframes[metric_12mo] = df_12mo
        self.new_classification_map[metric_12mo] = factor_category
        
        self.transform_stats['difference'] += 1
    
    def get_results(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """
        Get the transformation results.
        
        Returns:
        --------
        tuple
            (transformed_dataframes, new_classification_map)
        """
        if not self.transformed_dataframes:
            raise ValueError("No transformations performed. Run transform() first.")
        
        return self.transformed_dataframes, self.new_classification_map
    
    def validate(self, verbose: bool = True) -> 'QuantMetricsTransformer':
        """
        Validate the transformations.
        
        Parameters:
        -----------
        verbose : bool, default=True
            Whether to print detailed validation results
            
        Returns:
        --------
        self : QuantMetricsTransformer
            Returns self for method chaining (validation result stored in self.validation_passed)
        """
        if not self.transformed_dataframes:
            if verbose:
                print("No transformations to validate. Run transform() first.")
            self.validation_passed = False
            return self
        
        # Calculate expected metrics count
        original_metrics = [m for m in self.transformation_map.keys() 
                          if m in self.original_dataframes]
        
        expected_count = (
            sum(1 for m in original_metrics 
                if self.transformation_map[m] == 'absolute') +
            sum(2 for m in original_metrics 
                if self.transformation_map[m] == 'percent') +
            sum(2 for m in original_metrics 
                if self.transformation_map[m] == 'difference')
        )
        
        actual_count = len(self.transformed_dataframes)
        validation_passed = actual_count == expected_count
        
        # Store validation result as instance attribute
        self.validation_passed = validation_passed
        
        if verbose:
            print("=== TRANSFORMATION VALIDATION ===\n")
            print("Original metrics by transformation type:")
            for transform_type in ['absolute', 'percent', 'difference']:
                count = self.transform_stats[transform_type]
                print(f"  {transform_type}: {count} metrics")
            
            print(f"\nTotal original metrics processed: {sum(self.transform_stats.values()) - self.transform_stats['skipped']}")
            print(f"Total transformed metrics: {actual_count}")
            print(f"Expected transformed metrics: {expected_count}")
            print(f"Skipped metrics: {self.transform_stats['skipped']}")
            print(f"Validation: {'✓ PASSED' if validation_passed else '✗ FAILED'}")
        
        return self
    
    def generate_report(self) -> None:
        """Generate a comprehensive summary report of the transformations."""
        if not self.transformed_dataframes:
            print("No transformations to report. Run transform() first.")
            return
        
        print("\n=== TRANSFORMATION SUMMARY REPORT ===\n")
        
        # Group by factor category
        factor_groups = {}
        for metric, factor in self.new_classification_map.items():
            if factor not in factor_groups:
                factor_groups[factor] = []
            factor_groups[factor].append(metric)
        
        print("Transformed metrics by factor category:\n")
        for factor, metrics in sorted(factor_groups.items()):
            print(f"{factor.upper()} ({len(metrics)} metrics):")
            
            # Separate by transformation type
            absolute_metrics = [m for m in metrics if not ('_3mo_' in m or '_12mo_' in m)]
            three_mo_metrics = [m for m in metrics if '_3mo_' in m]
            twelve_mo_metrics = [m for m in metrics if '_12mo_' in m]
            
            if absolute_metrics:
                print(f"  Absolute: {', '.join(sorted(absolute_metrics))}")
            if three_mo_metrics:
                print(f"  3-Month: {', '.join(sorted(three_mo_metrics))}")
            if twelve_mo_metrics:
                print(f"  12-Month: {', '.join(sorted(twelve_mo_metrics))}")
            print()
    
    def get_metrics_by_factor(self, factor_category: str) -> Dict[str, pd.DataFrame]:
        """
        Get all transformed metrics for a specific factor category.
        
        Parameters:
        -----------
        factor_category : str
            The factor category to filter by (e.g., 'Valuation', 'Quality')
            
        Returns:
        --------
        dict
            Dictionary of metrics belonging to the specified factor category
        """
        if not self.transformed_dataframes:
            raise ValueError("No transformations performed. Run transform() first.")
        
        return {
            metric: df for metric, df in self.transformed_dataframes.items()
            if self.new_classification_map.get(metric) == factor_category
        }
    
    def get_metrics_by_transformation(self, transformation_type: str) -> Dict[str, pd.DataFrame]:
        """
        Get all metrics by transformation type.
        
        Parameters:
        -----------
        transformation_type : str
            The transformation type ('absolute', '3mo_pct_chg', '12mo_pct_chg', 
                                   '3mo_diff_chg', '12mo_diff_chg')
            
        Returns:
        --------
        dict
            Dictionary of metrics with the specified transformation type
        """
        if not self.transformed_dataframes:
            raise ValueError("No transformations performed. Run transform() first.")
        
        if transformation_type == 'absolute':
            return {
                metric: df for metric, df in self.transformed_dataframes.items()
                if not ('_3mo_' in metric or '_12mo_' in metric)
            }
        else:
            return {
                metric: df for metric, df in self.transformed_dataframes.items()
                if transformation_type in metric
            }
    
    def save_results(self, filepath: str, format: str = 'pickle') -> None:
        """
        Save transformation results to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the results
        format : str, default='pickle'
            Format to save ('pickle', 'excel', 'csv')
        """
        if not self.transformed_dataframes:
            raise ValueError("No transformations to save. Run transform() first.")
        
        if format == 'pickle':
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'transformed_dataframes': self.transformed_dataframes,
                    'classification_map': self.new_classification_map,
                    'transform_stats': self.transform_stats
                }, f)
        elif format == 'excel':
            with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
                for metric, df in self.transformed_dataframes.items():
                    # Excel sheet names have a 31 character limit
                    sheet_name = metric[:31] if len(metric) > 31 else metric
                    df.to_excel(writer, sheet_name=sheet_name)
        else:
            raise ValueError("Unsupported format. Use 'pickle' or 'excel'.")
        
        print(f"Results saved to {filepath}")


# Example usage demonstration
if __name__ == "__main__":
    
    # Create and configure the transformer
    transformer = QuantMetricsTransformer()
    
    # Example usage (assuming you have your dataframes ready)
    # transformer.load_data(dataFrames).transform().validate().generate_report()
    
    # Alternative usage with custom maps
    # custom_classification_map = {...}  # Your custom map
    # custom_transformation_map = {...}  # Your custom map
    # 
    # transformer = (QuantMetricsTransformer()
    #                .load_data(dataFrames)
    #                .set_classification_map(custom_classification_map)
    #                .set_transformation_map(custom_transformation_map)
    #                .transform()
    #                .validate()
    #                .generate_report())
    # 
    # # Get results
    # transformed_data, new_classification = transformer.get_results()
    # 
    # # Get specific factor metrics
    # valuation_metrics = transformer.get_metrics_by_factor('Valuation')
    # percent_change_metrics = transformer.get_metrics_by_transformation('12mo_pct_chg')
    # 
    # # Save results
    # transformer.save_results('transformed_metrics.pkl')
    
    print("QuantMetricsTransformer class ready for use!")
    print("\nBasic usage:")
    print("transformer = QuantMetricsTransformer()")
    print("transformer.load_data(dataFrames).transform().validate().generate_report()")
    print("transformed_data, classification_map = transformer.get_results()")



#%%
# Create and configure the transformer
transformer = QuantMetricsTransformer()
    
# Example usage (assuming you have your dataframes ready)
transformer.load_data(dataFrames).transform().validate().generate_report()

# # Get results
transformed_data, new_classification = transformer.get_results()


# Get specific factor metrics
valuation_metrics = transformer.get_metrics_by_factor('Valuation')
percent_change_metrics = transformer.get_metrics_by_transformation('3mo_pct_chg')

#%%

# Save results
transformer.save_results('transformed_metrics.excel')




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
        
        classification_map = {
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
        
        return classification_map
    
    def create_signal_directionality_map(self) -> Dict[str, int]:
        """
        Create a signal directionality map where 1 means "higher is better" and -1 means "lower is better".
        
        This ensures all factors are oriented consistently: higher values = more attractive investment.
        
        Returns:
        --------
        Dict[str, int] : Factor to directionality mapping (1 or -1)
        """
        
        directionality_map = {
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


# Example usage
if __name__ == "__main__":
    
    # Initialize framework for country analysis
    framework = CountryFactorSelectionFramework(
        correlation_threshold=0.85,
        vif_threshold=5.0,
        min_data_coverage=0.6
    )
    
    print("Country-Level Factor Selection Framework Ready!")
    print("\nFramework Features:")
    print("- Cross-sectional standardization across countries")
    print("- Hierarchical clustering with category awareness")
    print("- VIF-based multicollinearity detection")
    print("- Category balancing to ensure factor diversity")
    print("- Robust statistical methods for financial data")
    
    print(f"\nConfiguration:")
    print(f"- Correlation threshold: {framework.correlation_threshold}")
    print(f"- VIF threshold: {framework.vif_threshold}")
    print(f"- Minimum data coverage: {framework.min_data_coverage:.1%}")
    
    print("\nUsage Example:")
    print("# Run complete analysis")
    print("results = framework.run_complete_analysis(dataFrames)")
    print("")
    print("# Get selected factors")
    print("selected_factors = results['final_factors']")
    print("")
    print("# Create final modeling matrix")
    print("final_matrix = framework.create_final_factor_matrix(dataFrames, selected_factors)")
    print("")
    print("# Visualize correlations")
    print("framework.plot_correlation_heatmap(results['correlation_matrix'], selected_factors)")
    
    # Show classification preview
    classification_map = framework.create_classification_map()
    
    print(f"\nFactor Categories ({len(set(classification_map.values()))}):")
    category_preview = {}
    for factor, category in classification_map.items():
        if category not in category_preview:
            category_preview[category] = []
        category_preview[category].append(factor)
    
    for category, factors in category_preview.items():
        sample_factors = factors[:3]
        if len(factors) > 3:
            sample_factors.append(f"... (+{len(factors)-3} more)")
        print(f"  {category:<15}: {', '.join(sample_factors)}")
        

#%%

# Initialize framework
framework = CountryFactorSelectionFramework(
    correlation_threshold=0.85,
    vif_threshold=5.0,
    min_data_coverage=0.6  # 60% minimum data coverage
)

# Run complete analysis on your DataFrames
results = framework.run_complete_analysis(dataFrames)

# Get directionality report
directionality_report = framework.get_directionality_report()

# Get selected factors
selected_factors = results['final_factors']

directionality_corrected_dataFrames =  results['corrected_dataframes']


# Create final modeling matrix
final_matrix = framework.create_final_factor_matrix(directionality_corrected_dataFrames, selected_factors)

# Plot final correlation matrix
framework.plot_correlation_heatmap(results['correlation_matrix'], selected_factors)

# Export selected factors
export_selected_factors_data(dataFrames, selected_factors, 'SelectedFactors')

# Export selected factors - directionality corrected
export_selected_factors_data(directionality_corrected_dataFrames, selected_factors, 'SelectedFactorsDirectionalityCorrected')

# Validate selection quality
validation = validate_factor_selection(
    results['factor_matrix'], 
    selected_factors)

# Generate comprehensive report
create_factor_analysis_report(results, 'factor_analysis_report.txt')


classification_map= results['classification_map']


#%%

class CategorySignalGenerator:
    """
    A class to create category-specific composite signals from multiple investment factors.
    
    This class creates composite signals for each factor category by calculating
    the mean of z-scores, then re-standardizes these signals cross-sectionally
    for each date.
    """
    
    def __init__(self, data: pd.DataFrame, classification_map: Dict[str, str]):
        """
        Initialize the CategorySignalGenerator class.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Multi-index DataFrame with dates and countries as index,
            and standardized factors as columns
        classification_map : Dict[str, str]
            Dictionary mapping factor names to category names
            e.g., {'Assets': 'Quality', 'CashFlowYieldFWD': 'Valuation', ...}
        """
        self.data = data.copy()
        self.classification_map = classification_map
        
        # Convert classification_map to category-to-factors format for internal use
        self.category_to_factors = self._convert_classification_map()
        
        self.category_signals = {}
        
        # Validate inputs
        self._validate_inputs()
        
    def _convert_classification_map(self) -> Dict[str, List[str]]:
        """
        Convert classification_map from {factor: category} to {category: [factors]} format.
        
        Returns:
        --------
        Dict[str, List[str]] : Dictionary with category names as keys and 
                              lists of factor names as values
        """
        category_to_factors = {}
        
        for factor, category in self.classification_map.items():
            if category not in category_to_factors:
                category_to_factors[category] = []
            category_to_factors[category].append(factor)
            
        return category_to_factors
        
    def _validate_inputs(self):
        """Validate input data and classification map."""
        if not isinstance(self.data.index, pd.MultiIndex):
            raise ValueError("Data must have a MultiIndex with dates and countries")
            
        if len(self.data.index.names) != 2:
            raise ValueError("Data index must have exactly 2 levels (dates and countries)")
            
        # Get factors that exist in both classification_map and data
        mapped_factors = set(self.classification_map.keys())
        data_factors = set(self.data.columns)
        
        # Factors that are mapped and exist in data (these will be used)
        usable_factors = mapped_factors & data_factors
        
        # Factors in classification_map but not in data (expected - filtered out factors)
        missing_factors = mapped_factors - data_factors
        
        # Factors in data but not in classification_map (potential issue)
        unmapped_factors = data_factors - mapped_factors
        
        # Only raise error if no usable factors found
        if not usable_factors:
            raise ValueError("No factors from classification_map found in data. Check factor names.")
            
        # Warnings for informational purposes
        if missing_factors:
            print(f"Info: {len(missing_factors)} factors in classification_map not found in data (likely filtered out)")
            
        if unmapped_factors:
            print(f"Warning: {len(unmapped_factors)} factors in data not mapped to categories: {list(unmapped_factors)}")
            
        print(f"Validation successful:")
        print(f"  - {len(usable_factors)} usable factors mapped to {len(self.category_to_factors)} categories")
        print(f"  - Categories: {list(self.category_to_factors.keys())}")
        
        # Update category_to_factors to only include factors that exist in data
        self._filter_category_to_factors(data_factors)
    
    def _filter_category_to_factors(self, data_factors: set):
        """
        Filter category_to_factors to only include factors that exist in the data.
        
        Parameters:
        -----------
        data_factors : set
            Set of factor names that exist in the data
        """
        filtered_category_to_factors = {}
        
        for category, factors in self.category_to_factors.items():
            # Only keep factors that exist in the data
            existing_factors = [f for f in factors if f in data_factors]
            
            if existing_factors:  # Only keep categories that have at least one factor
                filtered_category_to_factors[category] = existing_factors
            else:
                print(f"Warning: Category '{category}' has no factors in the data - will be excluded")
        
        self.category_to_factors = filtered_category_to_factors
        
        print(f"After filtering:")
        for category, factors in self.category_to_factors.items():
            print(f"  - {category}: {len(factors)} factors")
    
    def create_category_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Create composite signals for each category by calculating the mean of factor z-scores.
        
        Note: Categories with only one factor will have that factor as their composite signal.
        
        Returns:
        --------
        Dict[str, pd.DataFrame] : Dictionary with category names as keys and 
                                  composite signals as values (in panel format)
        """
        print("Creating category composite signals...")
        
        for category, factors in self.category_to_factors.items():
            print(f"Processing category: {category}")
            
            # Select factors for this category
            category_factors = [f for f in factors if f in self.data.columns]
            
            if not category_factors:
                print(f"Warning: No valid factors found for category {category}")
                continue
                
            # Calculate equal-weighted composite (mean of z-scores)
            category_data = self.data[category_factors]
            
            if len(category_factors) == 1:
                print(f"  - Single factor category: using {category_factors[0]} directly")
                composite_signal = category_data.iloc[:, 0]  # Get the single factor
            else:
                # Handle missing values by taking mean of available factors
                composite_signal = category_data.mean(axis=1, skipna=True)
            
            # Convert to DataFrame for consistency
            composite_df = composite_signal.to_frame(name=f'{category}_Signal')
            
            # Reshape to panel format: dates as index, countries as columns
            panel_df = composite_df.unstack(level=1)  # Unstack the country level
            panel_df.columns = panel_df.columns.droplevel(0)  # Remove the signal name level
            panel_df.columns.name = None  # Remove column name
            
            # Store in dictionary
            self.category_signals[f'{category}_Signal'] = panel_df
            
            print(f"  - Created composite signal with {len(category_factors)} factors")
            print(f"  - Factors: {category_factors}")
            print(f"  - Panel shape: {panel_df.shape} (dates x countries)")
            print(f"  - Signal range: [{composite_signal.min():.4f}, {composite_signal.max():.4f}]")
        
        return self.category_signals
    
    def restandardize_category_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Re-standardize category signals cross-sectionally for each date.
        
        For each date and category:
        1. Subtract the cross-sectional mean (across countries)
        2. Divide by cross-sectional standard deviation
        
        Returns:
        --------
        Dict[str, pd.DataFrame] : Dictionary with restandardized category signals
        """
        print("Re-standardizing category signals cross-sectionally...")
        
        restandardized_signals = {}
        
        for category_name, signal_df in self.category_signals.items():
            print(f"Re-standardizing {category_name}...")
            
            # signal_df is already in panel format (dates x countries)
            # Apply cross-sectional standardization row by row (date by date)
            
            def standardize_row(row):
                """Standardize values within each row (across countries for each date)."""
                # Remove NaN values for calculation
                valid_values = row.dropna()
                
                if len(valid_values) <= 1:
                    # Not enough data for standardization, return zeros
                    return row * 0
                
                row_mean = valid_values.mean()
                row_std = valid_values.std()
                
                # Handle cases where std is 0 (all countries have same signal value)
                if row_std == 0 or pd.isna(row_std):
                    return row * 0  # Return zeros if no cross-sectional variation
                
                # Apply standardization only to non-NaN values
                standardized_row = row.copy()
                mask = ~row.isna()
                standardized_row[mask] = (row[mask] - row_mean) / row_std
                
                return standardized_row
            
            # Apply standardization to each row (date)
            restandardized_df = signal_df.apply(standardize_row, axis=1)
            
            # Store restandardized signal
            restandardized_signals[category_name] = restandardized_df
            self.category_signals[category_name] = restandardized_df  # Update original
            
            # Get some statistics for feedback
            all_values = restandardized_df.stack().dropna()
            print(f"  - Restandardized panel shape: {restandardized_df.shape} (dates x countries)")
            if len(all_values) > 0:
                print(f"  - Restandardized signal range: [{all_values.min():.4f}, {all_values.max():.4f}]")
            else:
                print(f"  - Warning: No valid values after restandardization")
        
        return restandardized_signals
    
    def generate_category_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Run the complete category signal generation pipeline.
        
        Returns:
        --------
        Dict[str, pd.DataFrame] : Category signals (restandardized) in panel format
        """
        print("="*60)
        print("GENERATING CATEGORY SIGNALS")
        print("="*60)
        
        # Step 1: Create category signals
        self.create_category_signals()
        
        # Step 2: Re-standardize category signals
        self.restandardize_category_signals()
        
        print("="*60)
        print("CATEGORY SIGNAL GENERATION COMPLETED")
        print("="*60)
        
        return self.category_signals
    
    def get_category_breakdown(self) -> pd.DataFrame:
        """
        Get a breakdown of factors by category (only factors available in data).
        
        Returns:
        --------
        pd.DataFrame : Breakdown showing which factors belong to each category
        """
        breakdown_list = []
        
        for category, factors in self.category_to_factors.items():
            for factor in factors:
                breakdown_list.append({
                    'Factor': factor,
                    'Category': category
                })
        
        return pd.DataFrame(breakdown_list).sort_values(['Category', 'Factor'])
    
    def get_all_factors_breakdown(self) -> pd.DataFrame:
        """
        Get a breakdown showing all factors from classification_map and their status.
        
        Returns:
        --------
        pd.DataFrame : Breakdown showing factor name, category, and availability status
        """
        breakdown_list = []
        data_factors = set(self.data.columns)
        
        for factor, category in self.classification_map.items():
            breakdown_list.append({
                'Factor': factor,
                'Category': category,
                'In_Data': factor in data_factors,
                'Status': 'Available' if factor in data_factors else 'Filtered_Out'
            })
        
        return pd.DataFrame(breakdown_list).sort_values(['Category', 'Status', 'Factor'])
    
    def get_signal_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for category signals.
        
        Returns:
        --------
        pd.DataFrame : Summary statistics for category signals
        """
        if not self.category_signals:
            raise ValueError("No category signals available. Run generate_category_signals() first.")
        
        stats_list = []
        
        # Category signals statistics
        for category_name, signal_df in self.category_signals.items():
            # For panel format, stack to get all values
            signal_values = signal_df.stack().dropna()
            stats = {
                'Signal': category_name,
                'Count': len(signal_values),
                'Mean': signal_values.mean(),
                'Std': signal_values.std(),
                'Min': signal_values.min(),
                'Max': signal_values.max(),
                'Skewness': signal_values.skew(),
                'Kurtosis': signal_values.kurtosis()
            }
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list).round(4)


class AlphaSignalGenerator:
    """
    A class to create final alpha signals from category signals.
    """
    
    def __init__(self, category_signals: Dict[str, pd.DataFrame]):
        """
        Initialize with category signals.
        
        Parameters:
        -----------
        category_signals : Dict[str, pd.DataFrame]
            Dictionary of category signals in panel format
        """
        self.category_signals = category_signals.copy()
        self.final_alpha_signal = None
        
    def create_final_alpha_signal(self, method: str = 'equal_weighted') -> pd.DataFrame:
        """
        Create final alpha signal by combining category signals.
        
        Parameters:
        -----------
        method : str
            Method to combine signals ('equal_weighted', 'weighted', etc.)
            
        Returns:
        --------
        pd.DataFrame : Final alpha signal in panel format
        """
        print("Creating final alpha signal...")
        
        if not self.category_signals:
            raise ValueError("No category signals available.")
        
        if method == 'equal_weighted':
            # Combine all category signals with equal weights
            combined_signals = pd.concat(self.category_signals.values(), axis=1, 
                                       keys=self.category_signals.keys())
            
            # Calculate equal-weighted average of category signals
            final_alpha = combined_signals.mean(axis=1, skipna=True)
            
            # Get countries from the category signals for consistent formatting
            sample_signal = list(self.category_signals.values())[0]
            countries = sample_signal.columns.tolist()
            
            # Expand final alpha to have same countries as columns (broadcasting)
            final_alpha_panel = pd.DataFrame(index=final_alpha.index, columns=countries)
            for country in countries:
                final_alpha_panel[country] = final_alpha.values
                
            self.final_alpha_signal = final_alpha_panel
        
        else:
            raise ValueError(f"Method '{method}' not implemented yet.")
        
        print(f"Final alpha signal created:")
        print(f"  - Panel shape: {self.final_alpha_signal.shape} (dates x countries)")
        print(f"  - Signal range: [{final_alpha.min():.4f}, {final_alpha.max():.4f}]")
        print(f"  - Signal mean: {final_alpha.mean():.4f}")
        print(f"  - Signal std: {final_alpha.std():.4f}")
        
        return self.final_alpha_signal


class SignalAnalyzer:
    """
    A class for analyzing and visualizing signals and their contributions.
    """
    
    def __init__(self, category_signals: Dict[str, pd.DataFrame], 
                 final_alpha_signal: pd.DataFrame = None):
        """
        Initialize with signals.
        
        Parameters:
        -----------
        category_signals : Dict[str, pd.DataFrame]
            Dictionary of category signals in panel format
        final_alpha_signal : pd.DataFrame, optional
            Final alpha signal in panel format
        """
        self.category_signals = category_signals.copy()
        self.final_alpha_signal = final_alpha_signal
        
    def calculate_category_contributions(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate how much each category contributes to the final alpha signal.
        """
        if self.final_alpha_signal is None:
            raise ValueError("No final alpha signal available.")
        
        print("Calculating category contributions to final alpha signal...")
        
        n_categories = len(self.category_signals)
        # Calculate individual contributions (each category contributes 1/N of its value)
        contributions = {}
        
        for category_name, signal_df in self.category_signals.items():
            # Each category contributes equally (1/N) to the final alpha
            contribution = signal_df / n_categories
            contributions[category_name.replace('_Signal', '_Contribution')] = contribution
        
        # Calculate statistics and verification
        contributions_df = pd.concat(contributions.values(), axis=1, keys=contributions.keys())

        # Verify that contributions sum to final alpha (within rounding error)
        reconstructed_alpha = pd.concat(contributions.values(), axis=1).sum(axis=1)
        
         # Calculate contribution statistics
        contribution_stats = []
        for contrib_name, contrib_df in contributions.items():
            # Stack to get all values for statistics
            contrib_values = contrib_df.stack().dropna()
            if len(contrib_values) > 0:
                stats = {
                    'Category': contrib_name.replace('_Contribution', ''),
                    'Mean_Contribution': contrib_values.mean(),
                    'Std_Contribution': contrib_values.std(),
                    'Min_Contribution': contrib_values.min(),
                    'Max_Contribution': contrib_values.max(),
                    'Abs_Mean_Contribution': contrib_values.abs().mean(),
                    'Weight_in_Final': 1/n_categories # Equal weighted
                }
                contribution_stats.append(stats)
        
        contribution_stats_df = pd.DataFrame(contribution_stats).round(4)

        # Calculate relative importance metrics
        importance_metrics = []
        
        # Get final alpha values for comparison
        final_alpha_values = self.final_alpha_signal.stack().dropna()
        
        for contrib_name, contrib_df in contributions.items():
            contrib_values = contrib_df.stack().dropna()
            
            if len(contrib_values) > 0 and len(final_alpha_values) > 0:
                # Align the series for correlation calculation
                aligned_contrib, aligned_alpha = contrib_values.align(final_alpha_values, join='inner')
                
                # Calculate correlation with final alpha
                correlation = aligned_contrib.corr(aligned_alpha) if len(aligned_contrib) > 1 else 0
                
                # Calculate average absolute contribution as % of average absolute alpha
                avg_abs_contrib = contrib_values.abs().mean()
                avg_abs_alpha = final_alpha_values.abs().mean()
                relative_magnitude = (avg_abs_contrib / avg_abs_alpha) * 100 if avg_abs_alpha != 0 else 0
                
                # Calculate variance contribution (how much this category's variance contributes to total)
                contrib_var = contrib_values.var()
                total_var = final_alpha_values.var()
                variance_contribution = (contrib_var / total_var) * 100 if total_var != 0 else 0
                
                importance = {
                    'Category': contrib_name.replace('_Contribution', ''),
                    'Correlation_with_Alpha': correlation,
                    'Avg_Abs_Contribution_pct': relative_magnitude,
                    'Variance_Contribution_pct': variance_contribution,
                    'Theoretical_Weight_pct': (1/n_categories) * 100
                }
                importance_metrics.append(importance)
        
        importance_df = pd.DataFrame(importance_metrics).round(4)

        # Verification
        alpha_reconstruction_error = (reconstructed_alpha - self.final_alpha_signal.iloc[:, 0]).abs().mean()
        
        print(f"Category contribution analysis completed:")
        print(f"  - {n_categories} categories contributing equally")
        print(f"  - Each category weight: {1/n_categories:.4f} ({100/n_categories:.2f}%)")
        print(f"  - Alpha reconstruction error: {alpha_reconstruction_error:.8f}")
        
        return {
            'contributions': contributions,
            'contributions_combined': contributions_df,
            'contribution_stats': contribution_stats_df,
            'relative_importance': importance_df,
            'reconstruction_error': alpha_reconstruction_error
        }

    def plot_category_contributions(self, date_range=None, countries=None, figsize=(15, 10)):
        """
        Plot category contributions over time and across countries.
        
        Parameters:
        -----------
        date_range : tuple, optional
            (start_date, end_date) to focus on specific period
        countries : list, optional
            List of countries to focus on
        figsize : tuple
            Figure size
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn are required for plotting")
            return
            
        contributions_analysis = self.calculate_category_contributions()
        contributions = contributions_analysis['contributions']
        
        # Filter data if requested
        if date_range:
            start_date, end_date = date_range
            contributions = {k: v.loc[start_date:end_date] for k, v in contributions.items()}
            
        if countries:
            contributions = {k: v[countries] for k, v in contributions.items()}
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Time series of average contributions
        avg_contributions_over_time = pd.DataFrame({
            k: v.mean(axis=1) for k, v in contributions.items()
        })
        
        axes[0, 0].plot(avg_contributions_over_time.index, avg_contributions_over_time.values)
        axes[0, 0].legend(avg_contributions_over_time.columns, bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].set_title('Average Category Contributions Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Average Contribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Distribution of contributions
        all_contrib_values = []
        contrib_labels = []
        
        for category, contrib_df in contributions.items():
            values = contrib_df.stack().dropna()
            all_contrib_values.extend(values.tolist())
            contrib_labels.extend([category.replace('_Contribution', '')] * len(values))
        
        contrib_data = pd.DataFrame({
            'Contribution': all_contrib_values,
            'Category': contrib_labels
        })
        
        sns.boxplot(data=contrib_data, x='Category', y='Contribution', ax=axes[0, 1])
        axes[0, 1].set_title('Distribution of Category Contributions')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Contribution correlation matrix
        contrib_corr_data = pd.DataFrame({
            k.replace('_Contribution', ''): v.stack().dropna() 
            for k, v in contributions.items()
        })
        
        corr_matrix = contrib_corr_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Category Contribution Correlations')
        
        # Plot 4: Final alpha vs sum of contributions (verification)
        final_alpha_values = self.final_alpha_signal.stack().dropna()
        
        # Calculate sum of contributions correctly
        # contributions is a dict of DataFrames, so we need to sum them properly
        contrib_dfs = list(contributions.values())
        sum_contributions_df = contrib_dfs[0].copy()
        for df in contrib_dfs[1:]:
            sum_contributions_df = sum_contributions_df.add(df, fill_value=0)
        
        sum_contributions = sum_contributions_df.stack().dropna()
        
        # Align the series
        aligned_alpha, aligned_sum = final_alpha_values.align(sum_contributions, join='inner')
        
        axes[1, 1].scatter(aligned_sum, aligned_alpha, alpha=0.5)
        axes[1, 1].plot([aligned_sum.min(), aligned_sum.max()], 
                       [aligned_sum.min(), aligned_sum.max()], 'r--', alpha=0.8)
        axes[1, 1].set_xlabel('Sum of Contributions')
        axes[1, 1].set_ylabel('Final Alpha Signal')
        axes[1, 1].set_title('Verification: Alpha vs Sum of Contributions')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr_coef = aligned_alpha.corr(aligned_sum)
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_coef:.6f}', 
                       transform=axes[1, 1].transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
        
        return contributions_analysis

    def plot_signal_distributions(self, figsize=(15, 10)):
        """
        Plot distributions of category signals and optionally final alpha signal.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting")
            return
        
        n_signals = len(self.category_signals) + (1 if self.final_alpha_signal is not None else 0)
        n_cols = min(3, n_signals)
        n_rows = (n_signals + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_signals == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        # Plot category signals
        for category_name, signal_df in self.category_signals.items():
            row_idx = plot_idx // n_cols
            col_idx = plot_idx % n_cols
            
            signal_values = signal_df.stack().dropna()
            
            if n_rows > 1:
                ax = axes[row_idx, col_idx]
            else:
                ax = axes[col_idx]
                
            ax.hist(signal_values, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{category_name}\nMean: {signal_values.mean():.3f}, Std: {signal_values.std():.3f}')
            ax.set_xlabel('Signal Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Plot final alpha signal
        if self.final_alpha_signal is not None:
            row_idx = plot_idx // n_cols
            col_idx = plot_idx % n_cols
            
            alpha_values = self.final_alpha_signal.stack().dropna()
            
            if n_rows > 1:
                ax = axes[row_idx, col_idx]
            else:
                ax = axes[col_idx]
                
            ax.hist(alpha_values, bins=50, alpha=0.7, color='red', edgecolor='black')
            ax.set_title(f'Final Alpha Signal\nMean: {alpha_values.mean():.3f}, Std: {alpha_values.std():.3f}')
            ax.set_xlabel('Signal Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()



#%%

# Initialize and run category generator
category_generator = CategorySignalGenerator(
    data=final_matrix,
    classification_map=classification_map
)

# Run the complete category signals
category_signals = category_generator.generate_category_signals()

# Get statistics
stats_df = category_generator.get_signal_statistics()
print(stats_df)


#%%

























#%%

#Create final alpha signal
alpha_generator = AlphaSignalGenerator(category_signals)
final_alpha = alpha_generator.create_final_alpha_signal(method='equal_weighted')

#%%

#: Analyze and visualize
analyzer = SignalAnalyzer(category_signals, final_alpha)
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

