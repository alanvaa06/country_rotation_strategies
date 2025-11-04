"""
Function Module for Country Rotation Strategy

This module contains all the utility functions for processing financial data,
regional classifications, and data transformations used in the country rotation strategy.
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


def read_excel_files_to_dict(folder_path):
    """
    Reads all Excel files (.xlsx and .xls) from a specified folder
    and returns a dictionary of DataFrames.

    Args:
        folder_path (str): The path to the folder containing the Excel files.

    Returns:
        dict: A dictionary where keys are the file names and values are
              the corresponding pandas DataFrames.
    """
    # Initialize an empty dictionary to store the DataFrames.
    excel_data = {}
    
    # Check if the provided folder path exists.
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return excel_data
    
    # Iterate through all files in the specified folder.
    for filename in os.listdir(folder_path):
        # Check if the file is an Excel file by its extension.
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            # Construct the full file path.
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Read the Excel file into a pandas DataFrame.
                # The file name is used as the key in the dictionary.
                print(f"Reading {filename}...")
                df = pd.read_excel(file_path, skiprows=2, index_col=0, parse_dates=True)
                key = filename.replace('.xlsx', '')
                excel_data[key] = df
            except Exception as e:
                # Handle potential errors during file reading.
                print(f"Could not read {filename}. Error: {e}")
                
    return excel_data


def get_regions_dict(classification):
    """
    Get a dictionary of regions from a classification DataFrame.
    
    Args:
        classification (pd.DataFrame): A DataFrame containing classification information.
        
    Returns:
        dict: A dictionary where keys are region names and values are lists of indices.
    """
    # Get the index of the classification DataFrame.
    classification = classification.copy()
    keys = ['DM', 'EM']
    regions_dict = {}
    for key in keys:
        regions_dict[key] = classification[classification.Segment == key].index.to_list()

    keys = ['Europe', 'Asia', 'LatAm']
    for key in keys:
        regions_dict[key] = classification[classification.Region == key].index.to_list()

    regions_dict['Global'] = classification.index.to_list()

    return regions_dict


def remove_weekends_optimized(dataframes_dict):
    """
    Optimized function to remove weekends from multiple DataFrames efficiently.
    
    Args:
        dataframes_dict (dict): Dictionary of DataFrames with datetime indices
        
    Returns:
        dict: Dictionary of DataFrames with weekends removed
    """
    processed_dict = {}
    
    for key, dataframe in dataframes_dict.items():
        try:
            # Skip empty DataFrames
            if dataframe.empty:
                processed_dict[key] = dataframe
                continue
                
            # Ensure datetime index (only convert if needed)
            if not isinstance(dataframe.index, pd.DatetimeIndex):
                dataframe.index = pd.to_datetime(dataframe.index)
            
            # Use vectorized boolean indexing for better performance
            # Monday=0, Sunday=6, so weekdays are 0-4
            weekday_mask = dataframe.index.dayofweek < 5
            processed_dict[key] = dataframe[weekday_mask].copy()
            
        except Exception as e:
            print(f"Warning: Could not process {key}: {e}")
            processed_dict[key] = dataframe  # Keep original if processing fails
    
    return processed_dict


def slice_data_frames_by_date(data_frames, target_date='2010-01-01', columns_to_drop=None):
    """
    Optimized function to slice multiple DataFrames by a target date and drop specified columns efficiently.
    
    Args:
        data_frames (dict): Dictionary containing pandas DataFrames with datetime indices
        target_date (str): Date to slice by in 'YYYY-MM-DD' format. Default: '2010-01-01'
        columns_to_drop (list): List of column names (countries) to drop from each DataFrame. 
                               Default: ['Saudi Arabia']
        
    Returns:
        dict: Dictionary containing sliced and filtered DataFrames
        
    Raises:
        ValueError: If target_date format is invalid
    """
    # Set default columns to drop
    if columns_to_drop is None:
        columns_to_drop = ['Saudi Arabia']
    
    # Convert target_date to pandas datetime once (more efficient)
    try:
        target_datetime = pd.to_datetime(target_date)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid target_date format '{target_date}'. Expected 'YYYY-MM-DD': {e}")
    
    sliced_data_frames = {}
    processed_count = 0
    dropped_columns_summary = {}
    
    print(f"Slicing {len(data_frames)} datasets from {target_date}...")
    if columns_to_drop:
        print(f"Dropping columns: {columns_to_drop}")
    
    for df_name, df in data_frames.items():
        try:
            # Skip empty DataFrames
            if df.empty:
                sliced_data_frames[df_name] = df
                continue
            
            # Drop specified columns if they exist
            columns_dropped = []
            if columns_to_drop:
                existing_columns = [col for col in columns_to_drop if col in df.columns]
                if existing_columns:
                    df = df.drop(columns=existing_columns)
                    columns_dropped = existing_columns
            
            dropped_columns_summary[df_name] = columns_dropped
            
            # Ensure datetime index (only convert if needed)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Check if target date is within the DataFrame's date range
            if df.index.max() < target_datetime:
                print(f"Warning: {df_name} ends before {target_date}, keeping empty DataFrame")
                sliced_data_frames[df_name] = df.iloc[0:0]  # Empty DataFrame with same structure
                continue
            
            # Use vectorized boolean indexing for better performance
            date_mask = df.index >= target_datetime
            sliced_df = df[date_mask].copy()
            
            # Store the sliced DataFrame
            sliced_data_frames[df_name] = sliced_df
            processed_count += 1
            
        except Exception as e:
            print(f"Warning: Could not slice {df_name}: {e}")
            sliced_data_frames[df_name] = df  # Keep original if slicing fails
    
    # Summary of dropped columns
    total_dropped = sum(len(cols) for cols in dropped_columns_summary.values())
    if total_dropped > 0:
        print(f"Dropped {total_dropped} column instances across all datasets")
    
    print(f"Successfully processed {processed_count}/{len(data_frames)} datasets")
    return sliced_data_frames


def transform_process_data(dataFrames: Dict[str, pd.DataFrame], 
                         classification: pd.DataFrame,
                         output_folder: str = 'ProcessedInputs') -> Dict[str, pd.DataFrame]:
    """
    Transform and process financial data for country-level macro and market analysis.
    
    This function performs comprehensive data transformations including:
    - Data cleaning and filling
    - Regional/segment aggregations
    - Financial metrics calculations
    - Yield and spread computations
    - Growth rate derivations
    - Rolling statistics
    
    Parameters:
    -----------
    dataFrames : Dict[str, pd.DataFrame]
        Dictionary containing financial time series data for different metrics
    classification : pd.DataFrame
        DataFrame containing country classifications by Region, Segment, and Type
    output_folder : str, default 'ProcessedInputs'
        Folder path to export processed data
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with original and transformed financial metrics
    """
    
    # Create a deep copy to avoid modifying original data
    processed_data = {key: df.copy() for key, df in dataFrames.items()}
    
    try:
        # ==========================================
        # 1. DATA CLEANING AND FILLING
        # ==========================================
        print("Step 1: Cleaning and filling missing data...")
        
        # Backward fill key fundamental metrics
        fundamental_metrics = ['EBITDA', 'EV_EBITDA', 'Fwd_EV_EBITDA', 'Net_Debt_Ebitda', 'PCF']
        for metric in fundamental_metrics:
            if metric in processed_data:
                processed_data[metric] = processed_data[metric].fillna(method='bfill')
        
        # ==========================================
        # 2. REGIONAL AND SEGMENT AGGREGATIONS
        # ==========================================
        print("Step 2: Computing regional and segment aggregations...")
        
        # Get classification indices
        em_countries = classification[classification['Segment'] == 'EM'].index.tolist()
        dm_countries = classification[classification['Segment'] == 'DM'].index.tolist()
        asia_countries = classification[classification['Region'] == 'Asia'].index.tolist()
        europe_countries = classification[classification['Region'] == 'Europe'].index.tolist()
        latam_countries = classification[classification['Region'] == 'LatAm'].index.tolist()
        world_indices = classification.index.tolist()
        
        # GDP Aggregations (Simple Sum)
        if 'GDP' in processed_data:
            processed_data['GDP']['EM'] = processed_data['GDP'][em_countries].sum(axis=1)
            processed_data['GDP']['DM'] = processed_data['GDP'][dm_countries].sum(axis=1)
        
        # M2 Money Supply Aggregations (Simple Sum)
        if 'M2' in processed_data:
            processed_data['M2']['DM'] = processed_data['M2'][dm_countries].sum(axis=1)
            processed_data['M2']['EM'] = processed_data['M2'][em_countries].sum(axis=1)
            processed_data['M2']['Europe'] = processed_data['M2'][europe_countries].sum(axis=1)
            processed_data['M2']['Asia'] = processed_data['M2'][asia_countries].sum(axis=1)
            processed_data['M2']['LatAm'] = processed_data['M2'][latam_countries].sum(axis=1)
            processed_data['M2']['World'] = processed_data['M2'][world_indices].sum(axis=1)
        
        # Market Cap Aggregations (Simple Sum)
        if 'Market_Cap' in processed_data:
            processed_data['Market_Cap']['DM'] = processed_data['Market_Cap'][dm_countries].sum(axis=1)
            processed_data['Market_Cap']['EM'] = processed_data['Market_Cap'][em_countries].sum(axis=1)
            processed_data['Market_Cap']['Europe'] = processed_data['Market_Cap'][europe_countries].sum(axis=1)
            processed_data['Market_Cap']['Asia'] = processed_data['Market_Cap'][asia_countries].sum(axis=1)
            processed_data['Market_Cap']['LatAm'] = processed_data['Market_Cap'][latam_countries].sum(axis=1)
        
        # 10-Year Bond Yields (Weighted Averages)
        if 'Ten_Year' in processed_data:
            # Helper function for weighted average calculation
            def weighted_average_bonds(countries_list, segment_name):
                if len(countries_list) > 0:
                    country_data = processed_data['Ten_Year'][countries_list]
                    weights = country_data.div(country_data.sum(axis=1), axis=0)
                    return (country_data * weights).sum(axis=1)
                return pd.Series(dtype=float)
            
            processed_data['Ten_Year'] = processed_data['Ten_Year'] / 100
            processed_data['Ten_Year']['DM'] = weighted_average_bonds(dm_countries, 'DM')
            processed_data['Ten_Year']['EM'] = weighted_average_bonds(em_countries, 'EM')
            processed_data['Ten_Year']['Asia'] = weighted_average_bonds(asia_countries, 'Asia')
            processed_data['Ten_Year']['Europe'] = weighted_average_bonds(europe_countries, 'Europe')
            processed_data['Ten_Year']['LatAm'] = weighted_average_bonds(latam_countries, 'LatAm')
            processed_data['Ten_Year']['World'] = weighted_average_bonds(world_indices, 'World')
        
        # ==========================================
        # 3. CONSENSUS GROWTH CALCULATIONS
        # ==========================================
        print("Step 3: Computing consensus growth metrics...")
        
        # Sales Growth
        if 'PS' in processed_data and 'Fwd_PS' in processed_data:
            processed_data['ConsensusSalesGrowth'] = processed_data['PS'] / processed_data['Fwd_PS'] - 1
        
        # EBITDA Growth
        if 'EV_EBITDA' in processed_data and 'Fwd_EV_EBITDA' in processed_data:
            processed_data['ConsensusEbitdaGrowth'] = processed_data['EV_EBITDA'] / processed_data['Fwd_EV_EBITDA'] - 1
        
        # Earnings Growth
        if 'PE' in processed_data and 'Fwd_PE' in processed_data:
            processed_data['ConsensusEarningsGrowth'] = processed_data['PE'] / processed_data['Fwd_PE'] - 1
        
        # Cash Flow Growth
        if 'PCF' in processed_data and 'Fwd_PCF' in processed_data:
            processed_data['ConsensusCashFlowGrowth'] = processed_data['PCF'] / processed_data['Fwd_PCF'] - 1
        
        # ==========================================
        # 4. YIELD CALCULATIONS
        # ==========================================
        print("Step 4: Computing yield metrics...")
        
        # Earnings Yields
        if 'PE' in processed_data:
            processed_data['EarningsYieldTTM'] = 1 / processed_data['PE']
        
        if 'Fwd_PE' in processed_data:
            processed_data['EarningsYieldFWD'] = 1 / processed_data['Fwd_PE']
        
        # Cash Flow Yields
        if 'PCF' in processed_data:
            processed_data['CashFlowYieldTTM'] = 1 / processed_data['PCF']
        
        if 'Fwd_PCF' in processed_data:
            processed_data['CashFlowYieldFWD'] = 1 / processed_data['Fwd_PCF']
        
        # ==========================================
        # 5. SPREAD CALCULATIONS (vs 10-Year Bonds)
        # ==========================================
        print("Step 5: Computing spread metrics...")
        
        if 'Ten_Year' in processed_data:
            # Earnings Yield Spreads
            if 'EarningsYieldTTM' in processed_data:
                processed_data['EarningsYieldTTMSpread'] = processed_data['EarningsYieldTTM'].sub(processed_data['Ten_Year'], fill_value=0)
            
            if 'EarningsYieldFWD' in processed_data:
                processed_data['EarningsYieldFWDSpread'] = processed_data['EarningsYieldFWD'].sub(processed_data['Ten_Year'], fill_value=0)
            
            # Cash Flow Yield Spreads
            if 'CashFlowYieldTTM' in processed_data:
                processed_data['CashFlowYieldTTMSpread'] = processed_data['CashFlowYieldTTM'].sub(processed_data['Ten_Year'], fill_value=0)
            
            if 'CashFlowYieldFWD' in processed_data:
                processed_data['CashFlowYieldFWDSpread'] = processed_data['CashFlowYieldFWD'].sub(processed_data['Ten_Year'], fill_value=0)
            
            # Dividend Yield Spreads
            if 'DVD' in processed_data:
                processed_data['DvdYieldTTMSpread'] = processed_data['DVD'].sub(processed_data['Ten_Year'], fill_value=0)
            
            if 'Fwd_DVD' in processed_data:
                processed_data['DvdYieldFWDSpread'] = processed_data['Fwd_DVD'].sub(processed_data['Ten_Year'], fill_value=0)
        
        # ==========================================
        # 6. FUNDAMENTAL CALCULATIONS
        # ==========================================
        print("Step 6: Computing fundamental metrics...")
        
        # Earnings Calculations
        if 'PE' in processed_data and 'Price' in processed_data:
            processed_data['Earnings'] = (1 / processed_data['PE']) * processed_data['Price']
        
        if 'EarningsYieldFWD' in processed_data and 'Price' in processed_data:
            processed_data['FwdEarnings'] = (processed_data['EarningsYieldFWD']) * processed_data['Price']
        
        
        if 'Fwd_PS' in processed_data and 'Price' in processed_data:
            processed_data['FwdRevenue'] = (1 / processed_data['Fwd_PS']) * processed_data['Price']
        
        # EBITDA Calculations
        if 'Fwd_EV_EBITDA' in processed_data and 'EV' in processed_data:
            processed_data['FwdEBITDA'] = (1 / processed_data['Fwd_EV_EBITDA']) * processed_data['EV']
        
        # Cash Flow Calculations
        if 'PCF' in processed_data and 'Price' in processed_data:
            processed_data['CF'] = (1 / processed_data['PCF']) * processed_data['Price']
        
        if 'Fwd_PCF' in processed_data and 'Price' in processed_data:
            processed_data['FwdCF'] = (1 / processed_data['Fwd_PCF']) * processed_data['Price']
        
        # ==========================================
        # 7. MARGIN CALCULATIONS
        # ==========================================
        print("Step 7: Computing margin metrics...")
        
        if 'Revenue' in processed_data:
            # EBIT Margin
            if 'EBIT' in processed_data:
                processed_data['EbitMargin'] = processed_data['EBIT'] / processed_data['Revenue']
            
            # EBITDA Margin
            if 'EBITDA' in processed_data:
                processed_data['EbitdaMargin'] = processed_data['EBITDA'] / processed_data['Revenue']
            
            # Net Margin
            if 'Earnings' in processed_data:
                processed_data['NetMargin'] = processed_data['Earnings'] / processed_data['Revenue']
            
            # Forward EBITDA Margin
            if 'FwdEBITDA' in processed_data:
                processed_data['FwdEBITDAMargin'] = processed_data['FwdEBITDA'] / processed_data['Revenue']
            
            # Forward Net Margin
            if 'FwdEarnings' in processed_data:
                processed_data['FwdNetMargin'] = processed_data['FwdEarnings'] / processed_data['Revenue']
        
        # ==========================================
        # 8. ROLLING STATISTICS
        # ==========================================
        print("Step 8: Computing rolling statistics...")
        
        # Rolling Earnings Growth (3-month average of quarterly growth)
        if 'Earnings' in processed_data:
            processed_data['RollingEarnings'] = processed_data['Earnings'].pct_change(63).rolling(21).mean()
        
        if 'FwdEarnings' in processed_data:
            processed_data['FwdRollingEarnings'] = processed_data['FwdEarnings'].pct_change(63).rolling(21).mean()
        
        # Rolling Volatility (3-month, annualized)
        if 'Price' in processed_data:
            processed_data['RollingVol'] = processed_data['Price'].pct_change().rolling(63).std() * np.sqrt(252)
        
        # Cumulative Flows (3-month sum)
        if 'Flows' in processed_data:
            processed_data['CumFlow'] = processed_data['Flows'].rolling(63).sum()
        
        # ==========================================
        # 9. BALANCE SHEET METRICS
        # ==========================================
        print("Step 9: Computing balance sheet metrics...")
        
        # Assets to Equity Ratio
        if 'Assets' in processed_data and 'Equity' in processed_data:
            processed_data['AssetsEquity'] = processed_data['Assets'] / processed_data['Equity']
        
        # ==========================================
        # 10. EXPORT TO FILES
        # ==========================================
        print(f"Step 10: Exporting {len(processed_data)} datasets to {output_folder}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Export each DataFrame to Excel
        for key, df in processed_data.items():
            try:
                output_path = os.path.join(output_folder, f"{key}.xlsx")
                df.to_excel(output_path, index=True)
                print(f"  ✓ Exported {key} to {output_path}")
            except Exception as e:
                warnings.warn(f"Failed to export {key}: {str(e)}")
        
        print(f"\n✓ Data transformation completed successfully!")
        print(f"✓ Created {len(processed_data)} processed datasets")
        print(f"✓ Files exported to: {output_folder}/")
        
        return processed_data
        
    except Exception as e:
        print(f"❌ Error during data transformation: {str(e)}")
        raise


def validate_inputs(dataFrames: Dict[str, pd.DataFrame], classification: pd.DataFrame) -> bool:
    """
    Validate input data structure and requirements.
    
    Parameters:
    -----------
    dataFrames : Dict[str, pd.DataFrame]
        Dictionary of financial data
    classification : pd.DataFrame
        Country classification data
        
    Returns:
    --------
    bool : True if validation passes
    """
    
    # Check if dataFrames is a dictionary
    if not isinstance(dataFrames, dict):
        raise ValueError("dataFrames must be a dictionary")
    
    # Check if classification has required columns
    required_cols = ['Segment', 'Region', 'Type']
    missing_cols = [col for col in required_cols if col not in classification.columns]
    if missing_cols:
        raise ValueError(f"classification missing required columns: {missing_cols}")
    
    # Check for minimum required data
    essential_metrics = ['Price', 'PE', 'Ten_Year']
    available_metrics = list(dataFrames.keys())
    missing_essential = [metric for metric in essential_metrics if metric not in available_metrics]
    
    if missing_essential:
        warnings.warn(f"Missing essential metrics: {missing_essential}. Some calculations may fail.")
    
    print(f"✓ Input validation passed")
    print(f"✓ Available metrics: {len(available_metrics)}")
    print(f"✓ Classification entries: {len(classification)}")
    
    return True


def load_classification_data(file_path='Classification.xlsx'):
    """
    Load classification data from Excel file.
    
    Args:
        file_path (str): Path to the classification Excel file
        
    Returns:
        tuple: (classification, classification_metricas, classification_map)
    """
    try:
        classification = pd.read_excel(file_path, index_col=0, sheet_name='regiones')
        classification_metricas = pd.read_excel(file_path, index_col=0, sheet_name='metricas')
        classification_map = classification_metricas.to_dict()['Block']
        
        return classification, classification_metricas, classification_map
    except Exception as e:
        print(f"Error loading classification data: {e}")
        raise

def explore_data(dataFrames, regions_dict):
    """
    Quick data exploration function for interactive analysis.
    
    Args:
        dataFrames (dict): Processed financial data
        regions_dict (dict): Regional classifications
    """
    print("\n" + "=" * 50)
    print("📊 QUICK DATA EXPLORATION")
    print("=" * 50)
    
    # Show available metrics
    print(f"\n📈 Available Metrics ({len(dataFrames)}):")
    metrics_list = list(dataFrames.keys())
    for i, metric in enumerate(metrics_list[:15], 1):  # Show first 15
        print(f"   {i:2d}. {metric}")
    if len(metrics_list) > 15:
        print(f"   ... and {len(metrics_list) - 15} more metrics")
    
    # Show regional breakdowns
    print(f"\n🌍 Regional Classifications:")
    for region, countries in regions_dict.items():
        print(f"   {region:8s}: {len(countries):2d} countries")
    
    # Show sample data ranges
    if 'Price' in dataFrames:
        price_data = dataFrames['Price']
        print(f"\n📅 Sample Data Range (Price):")
        print(f"   Start Date: {price_data.index.min().strftime('%Y-%m-%d')}")
        print(f"   End Date:   {price_data.index.max().strftime('%Y-%m-%d')}")
        print(f"   Countries:  {len(price_data.columns)}")
    
    print(f"\n✅ Data ready for quantitative analysis!")
    

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
    ONE_MONTH_DAYS = 21
    THREE_MONTH_DAYS = 63
    SIX_MONTH_DAYS = 126
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
            'ConsensusSalesGrowth': 'Profitability', 
            'ConsensusEbitdaGrowth': 'Profitability',
            'ConsensusEarningsGrowth': 'Profitability', 
            'ConsensusCashFlowGrowth': 'Profitability',
            
            # MOMENTUM FACTORS
            'RollingEarnings': 'Momentum', 
            'FwdRollingEarnings': 'Momentum',
            'CumFlow': 'Momentum', 
            'Flows': 'Momentum',
            
            # SIZE FACTORS
            'Market_Cap': 'Momentum', 
            'EV': 'Momentum', 
            'Revenue': 'Momentum',
            'FwdRevenue': 'Quality', 
            'Price': 'Momentum', 
            'Assets': 'Quality',
            
            # RISK FACTORS
            'RollingVol': 'Momentum', 
            'Ten_Year': 'Macro',
            
            # SENTIMENT FACTORS
            'SI': 'Momentum', 
            'SI_Ratio': 'Momentum',
            
            # MACRO FACTORS
            'GDP': 'Macro', 
            'M2': 'Macro'
        }
    
    @staticmethod
    def _get_default_transformation_map() -> Dict[str, str]:
        """Return the default transformation map."""
        return {
            # ABSOLUTE - No transformation needed
            'PE': 'percent', 
            'Fwd_PE': 'percent', 
            'PB': 'percent',
            'PS': 'percent', 
            'Fwd_PS': 'percent', 
            'PCF': 'percent',
            'Fwd_PCF': 'percent', 
            'EV_EBIT': 'percent', 
            'EV_EBITDA': 'percent',
            'Fwd_EV_EBITDA': 'percent', 
            'EarningsYieldTTM': 'difference',
            'EarningsYieldFWD': 'difference', 
            'CashFlowYieldTTM': 'difference',
            'CashFlowYieldFWD': 'difference', 
            'DVD': 'difference', 
            'Fwd_DVD': 'difference',
            'Debt_to_Equity': 'difference', 
            'Net_Debt_Ebitda': 'difference',
            'AssetsEquity': 'difference', 
            'SI': 'difference', 
            'SI_Ratio': 'difference',
            
            # DIFFERENCE - Point-to-point differences
            'EarningsYieldTTMSpread': 'difference', 
            'EarningsYieldFWDSpread': 'difference',
            'CashFlowYieldTTMSpread': 'difference', 
            'CashFlowYieldFWDSpread': 'difference',
            'DvdYieldTTMSpread': 'difference', 
            'DvdYieldFWDSpread': 'difference',
            'EbitMargin': 'difference', 
            'EbitdaMargin': 'difference',
            'NetMargin': 'difference', 
            'FwdEBITDAMargin': 'difference',
            'FwdNetMargin': 'difference', 
            'ROE': 'difference', 
            'Fwd_ROE': 'difference',
            'Return_Capital': 'difference', 
            'ConsensusSalesGrowth': 'difference',
            'ConsensusEbitdaGrowth': 'difference', 
            'ConsensusEarningsGrowth': 'difference',
            'ConsensusCashFlowGrowth': 'difference', 
            'RollingEarnings': 'difference',
            'FwdRollingEarnings': 'difference', 
            'RollingVol': 'difference',
            'Ten_Year': 'difference', 
            'CumFlow': 'difference', 
            'Flows': 'difference',
            
            # PERCENT - Percentage changes
            'Debt': 'percent', 
            'Equity': 'percent', 
            'Liabilities': 'percent',
            'CF': 'percent',
            'FwdCF': 'percent', 
            'EBIT': 'percent',
            'EBITDA': 'percent', 
            'FwdEBITDA': 'percent', 
            'EPS': 'percent',
            'Earnings': 'percent', 
            'FwdEarnings': 'percent', 
            'Market_Cap': 'percent',
            'EV': 'percent', 
            'Revenue': 'percent', 
            'FwdRevenue': 'percent',
            'Price': 'percent', 
            'Assets': 'percent', 
            'GDP': 'percent', 
            'M2': 'percent'
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
        df_sliced = df.iloc[slice_days:].copy() if len(df) >= slice_days else df.copy()
        self.transformed_dataframes[metric_name] = df_sliced
        self.new_classification_map[metric_name] = factor_category
        self.transform_stats['absolute'] += 1
    
    def _process_percent(self, metric_name: str, df: pd.DataFrame, factor_category: str, slice_days: int) -> None:
        """Process metrics with percent change transformation."""
        # Apply transformations on full dataset first
        # 1-month percent change
        metric_1mo = f"{metric_name}_1mo_pct_chg"
        df_1mo_full = df.pct_change(periods=self.ONE_MONTH_DAYS) * 100
        # Then slice
        df_1mo = df_1mo_full.iloc[slice_days:].copy() if len(df_1mo_full) >= slice_days else df_1mo_full.copy()
        self.transformed_dataframes[metric_1mo] = df_1mo
        self.new_classification_map[metric_1mo] = factor_category

        # 3-month percent change
        metric_3mo = f"{metric_name}_3mo_pct_chg"
        df_3mo_full = df.pct_change(periods=self.THREE_MONTH_DAYS) * 100
        # Then slice
        df_3mo = df_3mo_full.iloc[slice_days:].copy() if len(df_3mo_full) >= slice_days else df_3mo_full.copy()
        self.transformed_dataframes[metric_3mo] = df_3mo
        self.new_classification_map[metric_3mo] = factor_category
        
        # 6-month percent change
        metric_6mo = f"{metric_name}_6mo_pct_chg"
        df_6mo_full = df.pct_change(periods=self.SIX_MONTH_DAYS) * 100
        # Then slice
        df_6mo = df_6mo_full.iloc[slice_days:].copy() if len(df_6mo_full) >= slice_days else df_6mo_full.copy()
        self.transformed_dataframes[metric_6mo] = df_6mo
        self.new_classification_map[metric_6mo] = factor_category

        # 12-month percent change  
        metric_12mo = f"{metric_name}_12mo_pct_chg"
        df_12mo_full = df.pct_change(periods=self.TWELVE_MONTH_DAYS) * 100
        # Then slice
        df_12mo = df_12mo_full.iloc[slice_days:].copy() if len(df_12mo_full) >= slice_days else df_12mo_full.copy()
        self.transformed_dataframes[metric_12mo] = df_12mo
        self.new_classification_map[metric_12mo] = factor_category
        
        self.transform_stats['percent'] += 1
    
    def _process_difference(self, metric_name: str, df: pd.DataFrame, factor_category: str, slice_days: int) -> None:
        """Process metrics with difference transformation."""
        # Apply transformations on full dataset first
        # 1-month percent change
        metric_1mo = f"{metric_name}_1mo_diff_chg"
        df_1mo_full = df.diff(periods=self.ONE_MONTH_DAYS) * 100
        # Then slice
        df_1mo = df_1mo_full.iloc[slice_days:].copy() if len(df_1mo_full) >= slice_days else df_1mo_full.copy()
        self.transformed_dataframes[metric_1mo] = df_1mo
        self.new_classification_map[metric_1mo] = factor_category

        # 3-month difference
        metric_3mo = f"{metric_name}_3mo_diff_chg"
        df_3mo_full = df.diff(periods=self.THREE_MONTH_DAYS)
        # Then slice
        df_3mo = df_3mo_full.iloc[slice_days:].copy() if len(df_3mo_full) >= slice_days else df_3mo_full.copy()
        self.transformed_dataframes[metric_3mo] = df_3mo
        self.new_classification_map[metric_3mo] = factor_category
        
        # 6-month percent change
        metric_6mo = f"{metric_name}_6mo_diff_chg"
        df_6mo_full = df.diff(periods=self.SIX_MONTH_DAYS) * 100
        # Then slice
        df_6mo = df_6mo_full.iloc[slice_days:].copy() if len(df_6mo_full) >= slice_days else df_6mo_full.copy()
        self.transformed_dataframes[metric_6mo] = df_6mo
        self.new_classification_map[metric_6mo] = factor_category

        # 12-month difference
        metric_12mo = f"{metric_name}_12mo_diff_chg"
        df_12mo_full = df.diff(periods=self.TWELVE_MONTH_DAYS)
        # Then slice
        df_12mo = df_12mo_full.iloc[slice_days:].copy() if len(df_12mo_full) >= slice_days else df_12mo_full.copy()
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
            absolute_metrics = [m for m in metrics if not ('_1mo_' in m or '_3mo_' in m or '_6mo_' in m or '_12mo_' in m)]
            one_mo_metrics = [m for m in metrics if '_1mo_' in m]
            three_mo_metrics = [m for m in metrics if '_3mo_' in m]
            six_mo_metrics = [m for m in metrics if '_6mo_' in m]            
            twelve_mo_metrics = [m for m in metrics if '_12mo_' in m]
            
            if absolute_metrics:
                print(f"  Absolute: {', '.join(sorted(absolute_metrics))}")
            if one_mo_metrics:
                print(f"  1-Month: {', '.join(sorted(one_mo_metrics))}")    
            if three_mo_metrics:
                print(f"  3-Month: {', '.join(sorted(three_mo_metrics))}")
            if six_mo_metrics:
                print(f"  6-Month: {', '.join(sorted(six_mo_metrics))}")
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
                if not ('_1mo_' in metric or '_3mo_' in metric or '_6mo_' in metric or '_12mo_' in metric)
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
        
        # --- Add the overall title to the figure ---
        if countries:
            if len(countries)==1:
                country_names= countries[0]
                fig.suptitle(f'Category Contribution for {country_names}', fontsize=16, fontweight='bold')
            else:
                country_names=''
                for country in countries:
                    country_names= country_names+', '+country 
                    country_names=country_names[1:]
                #country_names= [country +',' for country in countries]
                fig.suptitle(f'Category Contribution for {country_names}', fontsize=16, fontweight='bold')


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