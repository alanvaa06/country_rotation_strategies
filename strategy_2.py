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
from typing import Dict, List, Tuple, Any
import warnings
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
from typing import Dict, List, Tuple, Any
from scipy.cluster.hierarchy import linkage, fcluster
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class CountryFactorSelectionFramework:
    """
    Advanced framework for factor selection and correlation filtering for country-level analysis.
    
    This class handles DataFrames dictionary where each key is a metric name and each DataFrame
    contains time series data across different countries. The analysis standardizes metrics
    across countries and performs factor selection at the country cross-section level.
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
            'earningsYieldTTM': 'Valuation',
            'earningsYieldFWD': 'Valuation',
            'cashFlowYieldTTM': 'Valuation',
            'cashFlowYieldFWD': 'Valuation',
            'DVD': 'Valuation',
            'Fwd_DVD': 'Valuation',
            
            # VALUATION SPREADS (Risk Premium)
            'earningsYieldTTMSpread': 'Valuation',
            'earningsYieldFWDSpread': 'Valuation',
            'cashFlowYieldTTMSpread': 'Valuation',
            'cashFlowYieldFWDSpread': 'Valuation',
            'dvdYieldTTMSpread': 'Valuation',
            'dvdYieldFWDSpread': 'Valuation',
            
            # QUALITY FACTORS
            'ROE': 'Quality',
            'Fwd_ROE': 'Quality',
            'Return_Capital': 'Quality',
            'Debt_to_Equity': 'Quality',
            'Net_Debt_Ebitda': 'Quality',
            'assetsEquity': 'Quality',
            'Assets': 'Quality',
            'Debt': 'Quality',
            'Equity': 'Quality',
            'Liabilities': 'Quality',
            'CF': 'Quality',
            'fwdCF': 'Quality',
            
            # PROFITABILITY FACTORS
            'ebitMargin': 'Profitability',
            'ebitdaMargin': 'Profitability', 
            'netMargin': 'Profitability',
            'fwdEBITDAMargin': 'Profitability',
            'fwdNetMargin': 'Profitability',
            'EBIT': 'Profitability',
            'EBITDA': 'Profitability',
            'fwdEBITDA': 'Profitability',
            'EPS': 'Profitability',
            'earnings': 'Profitability',
            'fwdEarnings': 'Profitability',
            
            # MOMENTUM FACTORS  
            'consensusSalesGrowth': 'Momentum',
            'consensusEbitdaGrowth': 'Momentum',
            'consensusEarningsGrowth': 'Momentum', 
            'consensusCashFlowGrowth': 'Momentum',
            'rollingEarnings': 'Momentum',
            'fwdRollingEarnings': 'Momentum',
            'cumFlow': 'Momentum',
            'Flows': 'Momentum',
            
            # SIZE FACTORS
            'Market_Cap': 'Size',
            'EV': 'Size',
            'Revenue': 'Size',
            'revenue': 'Size',
            'fwdRevenue': 'Size',
            'Price': 'Size',
            
            # RISK FACTORS
            'rollingVol': 'Risk',
            'Ten_Year': 'Risk',
            
            # SENTIMENT FACTORS
            'SI': 'Sentiment',
            'SI_Ratio': 'Sentiment',
            
            # MACRO FACTORS
            'GDP': 'Macro',
            'M2': 'Macro'
        }
        
        return classification_map
    
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
        print("Creating country-factor matrix...")
        print(f"Processing {len(dataFrames)} metrics...")
        
        if not dataFrames:
            raise ValueError("No metrics provided in dataFrames dictionary")
        
        # Step 1: Stack all DataFrames to create long-format data
        factor_panels = {}
        
        for metric_name, metric_df in dataFrames.items():
            if isinstance(metric_df, pd.DataFrame) and not metric_df.empty:
                print(f"  Processing {metric_name}: shape {metric_df.shape}")
                
                # Stack the DataFrame: (date, country) -> value
                try:
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
        print(f"Index levels: {combined_data.index.names}")
        print(f"Sample index values: {combined_data.index[:5].tolist()}")
        
        # Step 3: Cross-sectional standardization at each date
        print("Performing cross-sectional standardization...")
        
        standardized_data = pd.DataFrame(index=combined_data.index, 
                                       columns=combined_data.columns)
        
        # Get unique dates - handle both MultiIndex and regular index cases
        try:
            if isinstance(combined_data.index, pd.MultiIndex):
                # Assume first level is dates
                unique_dates = combined_data.index.get_level_values(0).unique()
                print(f"Found {len(unique_dates)} unique dates in MultiIndex")
            else:
                # If not MultiIndex, treat index as dates
                unique_dates = combined_data.index.unique()
                print(f"Found {len(unique_dates)} unique dates in regular index")
                
            dates_processed = 0
            for date in unique_dates:
                try:
                    if isinstance(combined_data.index, pd.MultiIndex):
                        date_data = combined_data.loc[date]
                    else:
                        date_data = combined_data.loc[[date]]
                    
                    if len(date_data) > 1:  # Need multiple countries for standardization
                        # Robust standardization using median and MAD
                        for col in date_data.columns:
                            col_data = date_data[col].dropna()
                            
                            if len(col_data) >= 2:  # Minimum countries for meaningful standardization
                                median_val = col_data.median()
                                mad_val = np.median(np.abs(col_data - median_val))
                                
                                if mad_val > 0:
                                    # Convert MAD to standard deviation equivalent
                                    if isinstance(combined_data.index, pd.MultiIndex):
                                        standardized_data.loc[date, col] = (col_data - median_val) / (1.4826 * mad_val)
                                    else:
                                        standardized_data.loc[[date], col] = (col_data - median_val) / (1.4826 * mad_val)
                                else:
                                    # If MAD is 0, use simple centering
                                    if isinstance(combined_data.index, pd.MultiIndex):
                                        standardized_data.loc[date, col] = col_data - median_val
                                    else:
                                        standardized_data.loc[[date], col] = col_data - median_val
                            else:
                                # Insufficient data for standardization - keep original
                                if isinstance(combined_data.index, pd.MultiIndex):
                                    standardized_data.loc[date, col] = date_data[col]
                                else:
                                    standardized_data.loc[[date], col] = date_data[col]
                                    
                        dates_processed += 1
                        if dates_processed % 50 == 0:
                            print(f"    Processed {dates_processed}/{len(unique_dates)} dates...")
                            
                except Exception as e:
                    print(f"    Warning: Error processing date {date}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error in standardization: {e}")
            # Fallback: simple standardization across entire dataset
            print("Falling back to simple standardization...")
            for col in combined_data.columns:
                col_data = combined_data[col].dropna()
                if len(col_data) > 0:
                    standardized_data[col] = (combined_data[col] - col_data.median()) / col_data.std()
        
        # Step 4: Clean up the standardized data
        final_data = standardized_data.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        print(f"✓ Created standardized matrix: {final_data.shape[0]} observations × {final_data.shape[1]} factors")
        
        if final_data.empty:
            print("WARNING: Final standardized matrix is empty!")
            print("Debug information:")
            print(f"  - Original combined data shape: {combined_data.shape}")
            print(f"  - Non-null values in combined data: {combined_data.notna().sum().sum()}")
            print(f"  - Standardized data shape: {standardized_data.shape}")
            print(f"  - Non-null values in standardized data: {standardized_data.notna().sum().sum()}")
            
            # Return a minimal valid matrix for debugging
            if not combined_data.empty:
                return combined_data.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Store metadata
        self.results['data_info'] = {
            'original_metrics': len(dataFrames),
            'qualifying_metrics': len(factor_panels),
            'final_observations': final_data.shape[0],
            'final_factors': final_data.shape[1],
            'coverage_stats': {metric: factor_panels[metric].notna().mean() 
                             for metric in factor_panels.keys()}
        }
        
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
        
        if correlation_matrix.empty:
            print("WARNING: Correlation matrix is empty, returning empty factor list")
            return []
        
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
        
        if not candidate_factors:
            print("  No candidate factors provided")
            return []
        
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            print("  ⚠️  statsmodels not available, skipping VIF analysis")
            return candidate_factors
        
        # Prepare clean data
        available_factors = [f for f in candidate_factors if f in factor_matrix.columns]
        if not available_factors:
            print("  ⚠️  No candidate factors found in factor matrix")
            return candidate_factors
            
        vif_data = factor_matrix[available_factors].dropna()
        
        if len(vif_data) < len(available_factors) * 5:
            print("  ⚠️  Insufficient data for VIF analysis")
            return available_factors
        
        remaining_factors = available_factors.copy()
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
        print(f"✓ VIF analysis: {len(available_factors)} → {len(remaining_factors)} factors")
        
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
        
        if not selected_factors:
            print("  No factors to balance")
            return []
        
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
        Run the complete country-level factor selection analysis.
        
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
        print("="*70)
        
        # Step 1: Create classification map
        classification_map = self.create_classification_map()
        
        # Step 2: Create and standardize country-factor matrix
        factor_matrix = self.create_country_factor_matrix(dataFrames)
        
        if factor_matrix.empty:
            print("ERROR: Factor matrix is empty - cannot proceed with analysis")
            return {
                'classification_map': classification_map,
                'factor_matrix': factor_matrix,
                'correlation_matrix': pd.DataFrame(),
                'original_factor_count': len(dataFrames),
                'final_factors': [],
                'final_factor_count': 0,
                'reduction_ratio': 0.0,
                'processing_details': self.results,
                'error': 'Empty factor matrix'
            }
        
        # Step 3: Calculate correlation matrix
        correlation_matrix = self.calculate_correlation_matrix(factor_matrix)
        
        if correlation_matrix.empty:
            print("ERROR: Correlation matrix is empty - returning original factors")
            return {
                'classification_map': classification_map,
                'factor_matrix': factor_matrix,
                'correlation_matrix': correlation_matrix,
                'original_factor_count': len(dataFrames),
                'final_factors': list(factor_matrix.columns),
                'final_factor_count': len(factor_matrix.columns),
                'reduction_ratio': 1.0,
                'processing_details': self.results,
                'error': 'Empty correlation matrix'
            }
        
        # Step 4: Hierarchical clustering selection
        clustered_factors = self.hierarchical_clustering_selection(correlation_matrix, classification_map)
        
        # Step 5: VIF analysis
        vif_filtered_factors = self.variance_inflation_factor_analysis(factor_matrix, clustered_factors)
        
        # Step 6: Category balancing
        final_factors = self.balance_factor_categories(vif_filtered_factors, classification_map)
        
        # Compile results
        results = {
            'classification_map': classification_map,
            'factor_matrix': factor_matrix,
            'correlation_matrix': correlation_matrix,
            'original_factor_count': len(dataFrames),
            'final_factors': final_factors,
            'final_factor_count': len(final_factors),
            'reduction_ratio': len(final_factors) / len(dataFrames) if len(dataFrames) > 0 else 0,
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
        
        # Data coverage info
        if 'data_info' in self.results:
            info = self.results['data_info']
            print(f"\nData Quality:")
            print(f"  Qualifying metrics: {info['qualifying_metrics']}/{info['original_metrics']}")
            print(f"  Total observations: {info['final_observations']:,}")
            if info['coverage_stats']:
                print(f"  Average coverage: {np.mean(list(info['coverage_stats'].values())):.1%}")
        
        # Category breakdown
        classification_map = results['classification_map']
        final_factors = results['final_factors']
        
        if final_factors:
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
        if final_factors:
            print(f"\nSelected Factors ({len(final_factors)}):")
            print("-" * 50)
            
            # Group by category for display
            categorized_factors = {}
            for factor in sorted(final_factors):
                category = classification_map.get(factor, 'Unknown')
                if category not in categorized_factors:
                    categorized_factors[category] = []
                categorized_factors[category].append(factor)
            
            for category in sorted(categorized_factors.keys()):
                print(f"\n{category}:")
                for i, factor in enumerate(categorized_factors[category], 1):
                    print(f"  {i}. {factor}")
        else:
            print(f"\nNo factors were selected!")
    
    def create_final_factor_matrix(self, dataFrames: Dict[str, pd.DataFrame], 
                                 selected_factors: List[str]) -> pd.DataFrame:
        """
        Create final standardized factor matrix with only selected factors.
        
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
        
        if not selected_factors:
            print("WARNING: No selected factors provided, returning empty DataFrame")
            return pd.DataFrame()
        
        # Filter dataFrames to selected factors only
        selected_data = {factor: dataFrames[factor] for factor in selected_factors 
                        if factor in dataFrames}
        
        if not selected_data:
            print("WARNING: None of the selected factors found in original dataFrames")
            return pd.DataFrame()
        
        # Create standardized matrix using same process
        final_matrix = self.create_country_factor_matrix(selected_data)
        
        print(f"✓ Final matrix shape: {final_matrix.shape}")
        return final_matrix
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                               selected_factors: List[str] = None,
                               figsize: Tuple[int, int] = (14, 12)) -> None:
        """
        Plot correlation heatmap for analysis.
        
        Parameters:
        -----------
        correlation_matrix : pd.DataFrame
            Factor correlation matrix
        selected_factors : List[str], optional
            Factors to highlight, if None uses all
        figsize : Tuple[int, int]
            Figure size
        """
        
        if correlation_matrix.empty:
            print("Cannot plot heatmap: correlation matrix is empty")
            return
        
        if selected_factors:
            # Filter to selected factors that exist in the matrix
            available_factors = [f for f in selected_factors if f in correlation_matrix.columns]
            if not available_factors:
                print("Cannot plot heatmap: no selected factors found in correlation matrix")
                return
            plot_matrix = correlation_matrix.loc[available_factors, available_factors]
            title = f'Correlation Matrix - Selected Factors ({len(available_factors)})'
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
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Factors', fontsize=12)
        plt.ylabel('Factors', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

# Diagnostic function to help debug data structure issues
def diagnose_dataframes_structure(dataFrames: Dict[str, pd.DataFrame]) -> None:
    """
    Diagnose the structure of dataFrames to identify potential issues.
    
    Parameters:
    -----------
    dataFrames : Dict[str, pd.DataFrame]
        Dictionary of DataFrames to diagnose
    """
    
    print("="*70)
    print("DATAFRAMES STRUCTURE DIAGNOSIS")
    print("="*70)
    
    if not dataFrames:
        print("ERROR: dataFrames dictionary is empty!")
        return
    
    print(f"Total metrics: {len(dataFrames)}")
    
    # Sample a few DataFrames for detailed inspection
    sample_keys = list(dataFrames.keys())[:3]
    
    for key in sample_keys:
        df = dataFrames[key]
        print(f"\nMetric: {key}")
        print(f"  Shape: {df.shape}")
        print(f"  Index type: {type(df.index)}")
        print(f"  Index name: {df.index.name}")
        print(f"  Column count: {len(df.columns)}")
        print(f"  Sample columns: {list(df.columns[:5])}")
        print(f"  Sample index: {df.index[:3].tolist()}")
        print(f"  Data types: {df.dtypes.value_counts().to_dict()}")
        print(f"  Missing values: {df.isna().sum().sum()} / {df.size}")
        
        # Check for empty or all-NaN data
        if df.empty:
            print("  WARNING: DataFrame is empty!")
        elif df.isna().all().all():
            print("  WARNING: DataFrame contains only NaN values!")
        
        # Sample data
        non_null_data = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        if not non_null_data.empty:
            print(f"  Non-null data shape: {non_null_data.shape}")
            print(f"  Sample values:\n{non_null_data.iloc[:2, :3]}")
        else:
            print("  WARNING: No non-null data found after cleanup!")
    
    # Check data consistency across metrics
    print(f"\nCONSISTENCY CHECK:")
    
    shapes = [df.shape for df in dataFrames.values() if isinstance(df, pd.DataFrame)]
    unique_shapes = set(shapes)
    print(f"  Unique shapes: {unique_shapes}")
    
    if len(unique_shapes) > 1:
        print("  WARNING: DataFrames have inconsistent shapes!")
        shape_counts = {}
        for shape in shapes:
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        for shape, count in shape_counts.items():
            print(f"    Shape {shape}: {count} DataFrames")
    
    # Check index consistency
    indices = []
    for key, df in list(dataFrames.items())[:5]:  # Check first 5
        if isinstance(df, pd.DataFrame) and not df.empty:
            indices.append((key, df.index))
    
    if len(indices) > 1:
        base_key, base_index = indices[0]
        consistent_indices = True
        for key, idx in indices[1:]:
            if not base_index.equals(idx):
                consistent_indices = False
                break
        
        if consistent_indices:
            print("  ✓ Indices are consistent across DataFrames")
        else:
            print("  WARNING: Indices are inconsistent across DataFrames!")
            for key, idx in indices:
                print(f"    {key}: {len(idx)} entries, type: {type(idx)}")

# Usage example with diagnostics
def run_factor_analysis_with_diagnostics(dataFrames: Dict[str, pd.DataFrame],
                                       correlation_threshold: float = 0.85,
                                       vif_threshold: float = 5.0,
                                       min_data_coverage: float = 0.6) -> Dict[str, Any]:
    """
    Run factor analysis with comprehensive diagnostics.
    
    Parameters:
    -----------
    dataFrames : Dict[str, pd.DataFrame]
        Dictionary of DataFrames to analyze
    correlation_threshold : float
        Correlation threshold for factor selection
    vif_threshold : float  
        VIF threshold for multicollinearity
    min_data_coverage : float
        Minimum data coverage threshold
        
    Returns:
    --------
    Dict[str, Any] : Analysis results
    """
    
    # Step 1: Diagnose data structure
    diagnose_dataframes_structure(dataFrames)
    
    # Step 2: Initialize and run framework
    framework = CountryFactorSelectionFramework(
        correlation_threshold=correlation_threshold,
        vif_threshold=vif_threshold,
        min_data_coverage=min_data_coverage
    )
    
    # Step 3: Run analysis
    try:
        results = framework.run_complete_analysis(dataFrames)
        
        # Step 4: Additional validation if successful
        if results['final_factors']:
            print(f"\n{'='*70}")
            print("RUNNING ADDITIONAL VALIDATION")
            print(f"{'='*70}")
            
            validation = validate_factor_selection(
                results['factor_matrix'], 
                results['final_factors'],
                correlation_threshold
            )
            results['validation'] = validation
        
        return results
        
    except Exception as e:
        print(f"\nERROR in factor analysis: {e}")
        print("This error suggests a data structure issue.")
        
        # Return diagnostic info
        return {
            'error': str(e),
            'diagnostic_info': 'Run diagnose_dataframes_structure() for more details'
        }

# Additional helper functions
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
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("QUANTITATIVE FACTOR SELECTION ANALYSIS REPORT\n")
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
        
        # Data Quality
        if 'data_info' in results['processing_details']:
            info = results['processing_details']['data_info']
            f.write("DATA QUALITY ASSESSMENT\n")
            f.write("-"*40 + "\n")
            f.write(f"Qualifying Metrics: {info['qualifying_metrics']}/{info['original_metrics']}\n")
            f.write(f"Total Observations: {info['final_observations']:,}\n")
            if info['coverage_stats']:
                f.write(f"Average Coverage: {np.mean(list(info['coverage_stats'].values())):.1%}\n\n")
        
        # Selected Factors by Category
        if results['final_factors']:
            classification_map = results['classification_map']
            final_factors = results['final_factors']
            
            categorized_factors = {}
            for factor in sorted(final_factors):
                category = classification_map.get(factor, 'Unknown')
                if category not in categorized_factors:
                    categorized_factors[category] = []
                categorized_factors[category].append(factor)
            
            f.write("SELECTED FACTORS BY CATEGORY\n")
            f.write("-"*40 + "\n")
            for category in sorted(categorized_factors.keys()):
                f.write(f"\n{category.upper()} ({len(categorized_factors[category])}):\n")
                for i, factor in enumerate(categorized_factors[category], 1):
                    f.write(f"  {i:2d}. {factor}\n")
        
        # Add other report sections...
        f.write(f"\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Comprehensive report saved to: {output_file}")

def validate_factor_selection(factor_matrix: pd.DataFrame, 
                            selected_factors: List[str],
                            correlation_threshold: float = 0.85) -> Dict[str, Any]:
    """
    Validate the quality of factor selection.
    """
    
    print("Validating factor selection quality...")
    
    if factor_matrix.empty or not selected_factors:
        return {
            'error': 'Empty factor matrix or no selected factors',
            'n_factors': 0,
            'n_observations': 0
        }
    
    # Get selected factor data
    available_factors = [f for f in selected_factors if f in factor_matrix.columns]
    selected_data = factor_matrix[available_factors].dropna()
    
    if selected_data.empty:
        return {
            'error': 'No data available for selected factors',
            'n_factors': len(available_factors),
            'n_observations': 0
        }
    
    # Calculate final correlations
    final_corr = selected_data.corr(method='spearman')
    
    # Check for remaining high correlations
    high_corr_pairs = []
    n_factors = len(available_factors)
    
    for i in range(n_factors):
        for j in range(i+1, n_factors):
            corr_val = final_corr.iloc[i, j]
            if abs(corr_val) >= correlation_threshold:
                high_corr_pairs.append({
                    'factor1': available_factors[i],
                    'factor2': available_factors[j],
                    'correlation': corr_val
                })
    
    # Calculate factor statistics
    factor_stats = {}
    for factor in available_factors:
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
        'n_factors': len(available_factors),
        'n_observations': len(selected_data),
        'high_correlations_remaining': len(high_corr_pairs),
        'max_abs_correlation': final_corr.abs().values[np.triu_indices_from(final_corr.values, k=1)].max() if len(final_corr) > 1 else 0,
        'avg_abs_correlation': final_corr.abs().values[np.triu_indices_from(final_corr.values, k=1)].mean() if len(final_corr) > 1 else 0,
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

#%%
# Use lower threshold
framework = CountryFactorSelectionFramework(
    correlation_threshold=0.85,
    vif_threshold=5.0,
    min_data_coverage=0.3  # Much lower threshold
)

results = framework.run_complete_analysis(dataFrames)

# Get selected factors
selected_factors = results['final_factors']

# Create final modeling matrix
final_matrix = framework.create_final_factor_matrix(dataFrames, selected_factors)

# Plot final correlation matrix
framework.plot_correlation_heatmap(results['correlation_matrix'])
#%%
# Export selected factors
export_selected_factors_data(dataFrames, selected_factors, 'SelectedFactors')

# Generate comprehensive report
create_factor_analysis_report(results, 'factor_analysis_report.txt')

# Validate selection quality
validation = validate_factor_selection(
    results['factor_matrix'], 
    selected_factors
)

#%%

