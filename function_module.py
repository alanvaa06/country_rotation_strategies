"""
Function Module for Country Rotation Strategy

This module contains all the utility functions for processing financial data,
regional classifications, and data transformations used in the country rotation strategy.
"""

import os
import pandas as pd
from typing import Dict, List, Any
import warnings
import numpy as np


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
            processed_data['consensusSalesGrowth'] = processed_data['PS'] / processed_data['Fwd_PS'] - 1
        
        # EBITDA Growth
        if 'EV_EBITDA' in processed_data and 'Fwd_EV_EBITDA' in processed_data:
            processed_data['consensusEbitdaGrowth'] = processed_data['EV_EBITDA'] / processed_data['Fwd_EV_EBITDA'] - 1
        
        # Earnings Growth
        if 'PE' in processed_data and 'Fwd_PE' in processed_data:
            processed_data['consensusEarningsGrowth'] = processed_data['PE'] / processed_data['Fwd_PE'] - 1
        
        # Cash Flow Growth
        if 'PCF' in processed_data and 'Fwd_PCF' in processed_data:
            processed_data['consensusCashFlowGrowth'] = processed_data['PCF'] / processed_data['Fwd_PCF'] - 1
        
        # ==========================================
        # 4. YIELD CALCULATIONS
        # ==========================================
        print("Step 4: Computing yield metrics...")
        
        # Earnings Yields
        if 'PE' in processed_data:
            processed_data['earningsYieldTTM'] = 1 / processed_data['PE']
        
        if 'Fwd_PE' in processed_data:
            processed_data['earningsYieldFWD'] = 1 / processed_data['Fwd_PE']
        
        # Cash Flow Yields
        if 'PCF' in processed_data:
            processed_data['cashFlowYieldTTM'] = 1 / processed_data['PCF']
        
        if 'Fwd_PCF' in processed_data:
            processed_data['cashFlowYieldFWD'] = 1 / processed_data['Fwd_PCF']
        
        # ==========================================
        # 5. SPREAD CALCULATIONS (vs 10-Year Bonds)
        # ==========================================
        print("Step 5: Computing spread metrics...")
        
        if 'Ten_Year' in processed_data:
            # Earnings Yield Spreads
            if 'earningsYieldTTM' in processed_data:
                processed_data['earningsYieldTTMSpread'] = processed_data['earningsYieldTTM'].sub(processed_data['Ten_Year'], fill_value=0)
            
            if 'earningsYieldFWD' in processed_data:
                processed_data['earningsYieldFWDSpread'] = processed_data['earningsYieldFWD'].sub(processed_data['Ten_Year'], fill_value=0)
            
            # Cash Flow Yield Spreads
            if 'cashFlowYieldTTM' in processed_data:
                processed_data['cashFlowYieldTTMSpread'] = processed_data['cashFlowYieldTTM'].sub(processed_data['Ten_Year'], fill_value=0)
            
            if 'cashFlowYieldFWD' in processed_data:
                processed_data['cashFlowYieldFWDSpread'] = processed_data['cashFlowYieldFWD'].sub(processed_data['Ten_Year'], fill_value=0)
            
            # Dividend Yield Spreads
            if 'DVD' in processed_data:
                processed_data['dvdYieldTTMSpread'] = processed_data['DVD'].sub(processed_data['Ten_Year'], fill_value=0)
            
            if 'Fwd_DVD' in processed_data:
                processed_data['dvdYieldFWDSpread'] = processed_data['Fwd_DVD'].sub(processed_data['Ten_Year'], fill_value=0)
        
        # ==========================================
        # 6. FUNDAMENTAL CALCULATIONS
        # ==========================================
        print("Step 6: Computing fundamental metrics...")
        
        # Earnings Calculations
        if 'PE' in processed_data and 'Price' in processed_data:
            processed_data['earnings'] = (1 / processed_data['PE']) * processed_data['Price']
        
        if 'earningsYieldFWD' in processed_data and 'Price' in processed_data:
            processed_data['fwdEarnings'] = (processed_data['earningsYieldFWD']) * processed_data['Price']
        
        # Revenue Calculations
        if 'PS' in processed_data and 'Price' in processed_data:
            processed_data['revenue'] = (1 / processed_data['PS']) * processed_data['Price']
        
        if 'Fwd_PS' in processed_data and 'Price' in processed_data:
            processed_data['fwdRevenue'] = (1 / processed_data['Fwd_PS']) * processed_data['Price']
        
        # EBITDA Calculations
        if 'Fwd_EV_EBITDA' in processed_data and 'EV' in processed_data:
            processed_data['fwdEBITDA'] = (1 / processed_data['Fwd_EV_EBITDA']) * processed_data['EV']
        
        # Cash Flow Calculations
        if 'PCF' in processed_data and 'Price' in processed_data:
            processed_data['CF'] = (1 / processed_data['PCF']) * processed_data['Price']
        
        if 'Fwd_PCF' in processed_data and 'Price' in processed_data:
            processed_data['fwdCF'] = (1 / processed_data['Fwd_PCF']) * processed_data['Price']
        
        # ==========================================
        # 7. MARGIN CALCULATIONS
        # ==========================================
        print("Step 7: Computing margin metrics...")
        
        if 'revenue' in processed_data:
            # EBIT Margin
            if 'EBIT' in processed_data:
                processed_data['ebitMargin'] = processed_data['EBIT'] / processed_data['revenue']
            
            # EBITDA Margin
            if 'EBITDA' in processed_data:
                processed_data['ebitdaMargin'] = processed_data['EBITDA'] / processed_data['revenue']
            
            # Net Margin
            if 'earnings' in processed_data:
                processed_data['netMargin'] = processed_data['earnings'] / processed_data['revenue']
            
            # Forward EBITDA Margin
            if 'fwdEBITDA' in processed_data:
                processed_data['fwdEBITDAMargin'] = processed_data['fwdEBITDA'] / processed_data['revenue']
            
            # Forward Net Margin
            if 'fwdEarnings' in processed_data:
                processed_data['fwdNetMargin'] = processed_data['fwdEarnings'] / processed_data['revenue']
        
        # ==========================================
        # 8. ROLLING STATISTICS
        # ==========================================
        print("Step 8: Computing rolling statistics...")
        
        # Rolling Earnings Growth (3-month average of quarterly growth)
        if 'earnings' in processed_data:
            processed_data['rollingEarnings'] = processed_data['earnings'].pct_change(63).rolling(21).mean()
        
        if 'fwdEarnings' in processed_data:
            processed_data['fwdRollingEarnings'] = processed_data['fwdEarnings'].pct_change(63).rolling(21).mean()
        
        # Rolling Volatility (3-month, annualized)
        if 'Price' in processed_data:
            processed_data['rollingVol'] = processed_data['Price'].pct_change().rolling(63).std() * np.sqrt(252)
        
        # Cumulative Flows (3-month sum)
        if 'Flows' in processed_data:
            processed_data['cumFlow'] = processed_data['Flows'].rolling(63).sum()
        
        # ==========================================
        # 9. BALANCE SHEET METRICS
        # ==========================================
        print("Step 9: Computing balance sheet metrics...")
        
        # Assets to Equity Ratio
        if 'Assets' in processed_data and 'Equity' in processed_data:
            processed_data['assetsEquity'] = processed_data['Assets'] / processed_data['Equity']
        
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