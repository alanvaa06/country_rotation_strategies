"""
ProcessData Module

This module provides a ProcessData class for loading, processing, and exporting
financial data for the country rotation strategy.

The class consolidates all data loading and processing functionality from the Inputs folder.

Author: Alan Vazquez, CFA, Master Jedi
Last Updated: November 2025
"""

import os
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')


class ProcessData:
    """
    A class for processing financial data for country rotation strategies.
    
    This class handles:
    - Loading Excel files from the Inputs folder
    - Loading classification data
    - Validating input data
    - Removing weekends from time series
    - Slicing data by date and filtering countries
    - Creating regional classifications
    - Transforming and processing financial metrics
    - Exporting processed data (optional)
    """
    
    def __init__(self, inputs_folder: str = 'Inputs', 
                 classification_file: str = 'Classification.xlsx',
                 target_date: str = '2010-01-01',
                 columns_to_drop: List[str] = None):
        """
        Initialize ProcessData instance.
        
        Args:
            inputs_folder: Path to folder containing input Excel files
            classification_file: Path to classification Excel file
            target_date: Target date for slicing data (default: '2010-01-01')
            columns_to_drop: List of columns (countries) to drop (default: ['Saudi Arabia'])
        """
        self.inputs_folder = inputs_folder
        self.classification_file = classification_file
        self.target_date = target_date
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else ['Saudi Arabia']
        
        # Data attributes (set during processing)
        self.data_frames = None
        self.original_file_names = None
        self.classification = None
        self.classification_metricas = None
        self.classification_map = None
        self.regions_dict = None
        self.processed_data = None
    
    def read_excel_files_to_dict(self, folder_path: str = None) -> Dict[str, pd.DataFrame]:
        """
        Reads all Excel files (.xlsx and .xls) from a specified folder
        and returns a dictionary of DataFrames.

        Args:
            folder_path: The path to the folder containing the Excel files.
                         If None, uses self.inputs_folder.

        Returns:
            dict: A dictionary where keys are the file names and values are
                  the corresponding pandas DataFrames.
        """
        if folder_path is None:
            folder_path = self.inputs_folder
        
        excel_data = {}
        
        if not os.path.exists(folder_path):
            print(f"Error: The folder '{folder_path}' does not exist.")
            return excel_data
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    print(f"Reading {filename}...")
                    df = pd.read_excel(file_path, skiprows=2, index_col=0, parse_dates=True)
                    key = filename.replace('.xlsx', '').replace('.xls', '')
                    excel_data[key] = df
                except Exception as e:
                    print(f"Could not read {filename}. Error: {e}")
        
        self.data_frames = excel_data
        return excel_data

    def load_classification_data(self, file_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load classification data from Excel file.
        
        Args:
            file_path: Path to the classification Excel file. If None, uses self.classification_file.
        
        Returns:
            tuple: (classification, classification_metricas, classification_map)
        """
        if file_path is None:
            file_path = self.classification_file
        
        try:
            classification = pd.read_excel(file_path, index_col=0, sheet_name='regiones')
            classification_metricas = pd.read_excel(file_path, index_col=0, sheet_name='metricas')
            classification_map = classification_metricas.to_dict()['Block']
            
            self.classification = classification
            self.classification_metricas = classification_metricas
            self.classification_map = classification_map
            
            return classification, classification_metricas, classification_map
        except Exception as e:
            print(f"Error loading classification data: {e}")
            raise

    def validate_inputs(self, data_frames: Dict[str, pd.DataFrame] = None, 
                       classification: pd.DataFrame = None) -> bool:
        """
        Validate input data structure and requirements.
        
        Args:
            data_frames: Dictionary of financial data. If None, uses self.data_frames.
            classification: Country classification data. If None, uses self.classification.
        
        Returns:
            bool: True if validation passes
        """
        if data_frames is None:
            data_frames = self.data_frames
        if classification is None:
            classification = self.classification
        
        if not isinstance(data_frames, dict):
            raise ValueError("dataFrames must be a dictionary")
        
        required_cols = ['Segment', 'Region', 'Type']
        missing_cols = [col for col in required_cols if col not in classification.columns]
        if missing_cols:
            raise ValueError(f"classification missing required columns: {missing_cols}")
        
        essential_metrics = ['Price', 'PE', 'Ten_Year']
        available_metrics = list(data_frames.keys())
        missing_essential = [metric for metric in essential_metrics if metric not in available_metrics]
        
        if missing_essential:
            warnings.warn(f"Missing essential metrics: {missing_essential}. Some calculations may fail.")
        
        print(f"Input validation passed")
        print(f"Available metrics: {len(available_metrics)}")
        print(f"Classification entries: {len(classification)}")
        
        return True

    def remove_weekends_optimized(self, dataframes_dict: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Optimized function to remove weekends from multiple DataFrames efficiently.
        
        Args:
            dataframes_dict: Dictionary of DataFrames with datetime indices.
                             If None, uses self.data_frames.
        
        Returns:
            dict: Dictionary of DataFrames with weekends removed
        """
        if dataframes_dict is None:
            dataframes_dict = self.data_frames
        
        processed_dict = {}
        
        for key, dataframe in dataframes_dict.items():
            try:
                if dataframe.empty:
                    processed_dict[key] = dataframe
                    continue
                
                if not isinstance(dataframe.index, pd.DatetimeIndex):
                    dataframe.index = pd.to_datetime(dataframe.index)
                
                weekday_mask = dataframe.index.dayofweek < 5
                processed_dict[key] = dataframe[weekday_mask].copy()
                
            except Exception as e:
                print(f"Warning: Could not process {key}: {e}")
                processed_dict[key] = dataframe
        
        self.data_frames = processed_dict
        return processed_dict

    def slice_data_frames_by_date(self, data_frames: Dict[str, pd.DataFrame] = None,
                                  target_date: str = None,
                                  columns_to_drop: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Optimized function to slice multiple DataFrames by a target date and drop specified columns efficiently.
        
        Args:
            data_frames: Dictionary containing pandas DataFrames with datetime indices.
                         If None, uses self.data_frames.
            target_date: Date to slice by in 'YYYY-MM-DD' format. If None, uses self.target_date.
            columns_to_drop: List of column names (countries) to drop. If None, uses self.columns_to_drop.
        
        Returns:
            dict: Dictionary containing sliced and filtered DataFrames
        """
        if data_frames is None:
            data_frames = self.data_frames
        if target_date is None:
            target_date = self.target_date
        if columns_to_drop is None:
            columns_to_drop = self.columns_to_drop
        
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
                if df.empty:
                    sliced_data_frames[df_name] = df
                    continue
                
                columns_dropped = []
                if columns_to_drop:
                    existing_columns = [col for col in columns_to_drop if col in df.columns]
                    if existing_columns:
                        df = df.drop(columns=existing_columns)
                        columns_dropped = existing_columns
                
                dropped_columns_summary[df_name] = columns_dropped
                
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                if df.index.max() < target_datetime:
                    print(f"Warning: {df_name} ends before {target_date}, keeping empty DataFrame")
                    sliced_data_frames[df_name] = df.iloc[0:0]
                    continue
                
                date_mask = df.index >= target_datetime
                sliced_df = df[date_mask].copy()
                
                sliced_data_frames[df_name] = sliced_df
                processed_count += 1
                
            except Exception as e:
                print(f"Warning: Could not slice {df_name}: {e}")
                sliced_data_frames[df_name] = df
        
        total_dropped = sum(len(cols) for cols in dropped_columns_summary.values())
        if total_dropped > 0:
            print(f"Dropped {total_dropped} column instances across all datasets")
        
        print(f"Successfully processed {processed_count}/{len(data_frames)} datasets")
        self.data_frames = sliced_data_frames
        return sliced_data_frames

    def get_regions_dict(self, classification: pd.DataFrame = None) -> Dict[str, List]:
        """
        Get a dictionary of regions from a classification DataFrame.
        
        Args:
            classification: A DataFrame containing classification information.
                            If None, uses self.classification.
        
        Returns:
            dict: A dictionary where keys are region names and values are lists of indices.
        """
        if classification is None:
            classification = self.classification
        
        classification = classification.copy()
        keys = ['DM', 'EM']
        regions_dict = {}
        for key in keys:
            regions_dict[key] = classification[classification.Segment == key].index.to_list()

        keys = ['Europe', 'Asia', 'LatAm']
        for key in keys:
            regions_dict[key] = classification[classification.Region == key].index.to_list()

        regions_dict['Global'] = classification.index.to_list()
        
        self.regions_dict = regions_dict
        return regions_dict

    def transform_process_data(self, 
                              data_frames: Dict[str, pd.DataFrame] = None,
                              classification: pd.DataFrame = None,
                              output_folder: str = 'outputs/processed_inputs',
                              export_data: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Transform and process financial data for country-level macro and market analysis.
        
        This function performs comprehensive data transformations including:
        - Data cleaning and filling
        - Regional/segment aggregations
        - Financial metrics calculations
        - Yield and spread computations
        - Growth rate derivations
        - Rolling statistics
        
        Args:
            data_frames: Dictionary containing financial time series data for different metrics.
                         If None, uses self.data_frames.
            classification: DataFrame containing country classifications by Region, Segment, and Type.
                            If None, uses self.classification.
            output_folder: Folder path to export processed data (default: 'outputs/processed_inputs').
                           Only used if export_data=True.
            export_data: Whether to export processed data to files (default: True).
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with original and transformed financial metrics
        """
        if data_frames is None:
            data_frames = self.data_frames
        if classification is None:
            classification = self.classification
        
        processed_data = {key: df.copy() for key, df in data_frames.items()}
        
        try:
            # ==========================================
            # 1. DATA CLEANING AND FILLING
            # ==========================================
            print("Step 1: Cleaning and filling missing data...")
            
            fundamental_metrics = ['EBITDA', 'EV_EBITDA', 'Fwd_EV_EBITDA', 'Net_Debt_Ebitda', 'PCF']
            for metric in fundamental_metrics:
                if metric in processed_data:
                    processed_data[metric] = processed_data[metric].bfill()
            
            # ==========================================
            # 2. REGIONAL AND SEGMENT AGGREGATIONS
            # ==========================================
            print("Step 2: Computing regional and segment aggregations...")
            
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
            
            if 'PS' in processed_data and 'Fwd_PS' in processed_data:
                processed_data['ConsensusSalesGrowth'] = processed_data['PS'] / processed_data['Fwd_PS'] - 1
            
            if 'EV_EBITDA' in processed_data and 'Fwd_EV_EBITDA' in processed_data:
                processed_data['ConsensusEbitdaGrowth'] = processed_data['EV_EBITDA'] / processed_data['Fwd_EV_EBITDA'] - 1
            
            if 'PE' in processed_data and 'Fwd_PE' in processed_data:
                processed_data['ConsensusEarningsGrowth'] = processed_data['PE'] / processed_data['Fwd_PE'] - 1
            
            if 'PCF' in processed_data and 'Fwd_PCF' in processed_data:
                processed_data['ConsensusCashFlowGrowth'] = processed_data['PCF'] / processed_data['Fwd_PCF'] - 1
            
            # ==========================================
            # 4. YIELD CALCULATIONS
            # ==========================================
            print("Step 4: Computing yield metrics...")
            
            if 'PE' in processed_data:
                processed_data['EarningsYieldTTM'] = 1 / processed_data['PE']
            
            if 'Fwd_PE' in processed_data:
                processed_data['EarningsYieldFWD'] = 1 / processed_data['Fwd_PE']
            
            if 'PCF' in processed_data:
                processed_data['CashFlowYieldTTM'] = 1 / processed_data['PCF']
            
            if 'Fwd_PCF' in processed_data:
                processed_data['CashFlowYieldFWD'] = 1 / processed_data['Fwd_PCF']
            
            # ==========================================
            # 5. SPREAD CALCULATIONS (vs 10-Year Bonds)
            # ==========================================
            print("Step 5: Computing spread metrics...")
            
            if 'Ten_Year' in processed_data:
                if 'EarningsYieldTTM' in processed_data:
                    processed_data['EarningsYieldTTMSpread'] = processed_data['EarningsYieldTTM'].sub(processed_data['Ten_Year'], fill_value=0)
                
                if 'EarningsYieldFWD' in processed_data:
                    processed_data['EarningsYieldFWDSpread'] = processed_data['EarningsYieldFWD'].sub(processed_data['Ten_Year'], fill_value=0)
                
                if 'CashFlowYieldTTM' in processed_data:
                    processed_data['CashFlowYieldTTMSpread'] = processed_data['CashFlowYieldTTM'].sub(processed_data['Ten_Year'], fill_value=0)
                
                if 'CashFlowYieldFWD' in processed_data:
                    processed_data['CashFlowYieldFWDSpread'] = processed_data['CashFlowYieldFWD'].sub(processed_data['Ten_Year'], fill_value=0)
                
                if 'DVD' in processed_data:
                    processed_data['DvdYieldTTMSpread'] = processed_data['DVD'].sub(processed_data['Ten_Year'], fill_value=0)
                
                if 'Fwd_DVD' in processed_data:
                    processed_data['DvdYieldFWDSpread'] = processed_data['Fwd_DVD'].sub(processed_data['Ten_Year'], fill_value=0)
            
            # ==========================================
            # 6. FUNDAMENTAL CALCULATIONS
            # ==========================================
            print("Step 6: Computing fundamental metrics...")
            
            if 'PE' in processed_data and 'Price' in processed_data:
                processed_data['Earnings'] = (1 / processed_data['PE']) * processed_data['Price']
            
            if 'EarningsYieldFWD' in processed_data and 'Price' in processed_data:
                processed_data['FwdEarnings'] = (processed_data['EarningsYieldFWD']) * processed_data['Price']
            
            if 'Fwd_PS' in processed_data and 'Price' in processed_data:
                processed_data['FwdRevenue'] = (1 / processed_data['Fwd_PS']) * processed_data['Price']
            
            if 'Fwd_EV_EBITDA' in processed_data and 'EV' in processed_data:
                processed_data['FwdEBITDA'] = (1 / processed_data['Fwd_EV_EBITDA']) * processed_data['EV']
            
            if 'PCF' in processed_data and 'Price' in processed_data:
                processed_data['CF'] = (1 / processed_data['PCF']) * processed_data['Price']
            
            if 'Fwd_PCF' in processed_data and 'Price' in processed_data:
                processed_data['FwdCF'] = (1 / processed_data['Fwd_PCF']) * processed_data['Price']
            
            # ==========================================
            # 7. MARGIN CALCULATIONS
            # ==========================================
            print("Step 7: Computing margin metrics...")
            
            if 'Revenue' in processed_data:
                if 'EBIT' in processed_data:
                    processed_data['EbitMargin'] = processed_data['EBIT'] / processed_data['Revenue']
                
                if 'EBITDA' in processed_data:
                    processed_data['EbitdaMargin'] = processed_data['EBITDA'] / processed_data['Revenue']
                
                if 'Earnings' in processed_data:
                    processed_data['NetMargin'] = processed_data['Earnings'] / processed_data['Revenue']
                
                if 'FwdEBITDA' in processed_data:
                    processed_data['FwdEBITDAMargin'] = processed_data['FwdEBITDA'] / processed_data['Revenue']
                
                if 'FwdEarnings' in processed_data:
                    processed_data['FwdNetMargin'] = processed_data['FwdEarnings'] / processed_data['Revenue']
            
            # ==========================================
            # 8. ROLLING STATISTICS
            # ==========================================
            print("Step 8: Computing rolling statistics...")
            
            if 'Earnings' in processed_data:
                processed_data['RollingEarnings'] = processed_data['Earnings'].pct_change(63).rolling(21).mean()
            
            if 'FwdEarnings' in processed_data:
                processed_data['FwdRollingEarnings'] = processed_data['FwdEarnings'].pct_change(63).rolling(21).mean()
            
            if 'Price' in processed_data:
                processed_data['RollingVol'] = processed_data['Price'].pct_change().rolling(63).std() * np.sqrt(252)
            
            if 'Flows' in processed_data:
                processed_data['CumFlow'] = processed_data['Flows'].rolling(63).sum()
            
            # ==========================================
            # 9. BALANCE SHEET METRICS
            # ==========================================
            print("Step 9: Computing balance sheet metrics...")
            
            if 'Assets' in processed_data and 'Equity' in processed_data:
                processed_data['AssetsEquity'] = processed_data['Assets'] / processed_data['Equity']
            
            # ==========================================
            # 10. EXPORT TO FILES (Optional)
            # ==========================================
            if export_data:
                print(f"Step 10: Exporting {len(processed_data)} datasets to {output_folder}...")
                
                os.makedirs(output_folder, exist_ok=True)
                
                for key, df in processed_data.items():
                    try:
                        output_path = os.path.join(output_folder, f"{key}.xlsx")
                        df.to_excel(output_path, index=True)
                        print(f"  Exported {key} to {output_path}")
                    except Exception as e:
                        warnings.warn(f"Failed to export {key}: {str(e)}")
                
                print(f"\nData transformation completed successfully!")
                print(f"Created {len(processed_data)} processed datasets")
                print(f"Files exported to: {output_folder}/")
            else:
                print(f"\nData transformation completed successfully!")
                print(f"Created {len(processed_data)} processed datasets")
                print(f"Export skipped (export_data=False)")
            
            self.processed_data = processed_data
            return processed_data
            
        except Exception as e:
            print(f"Error during data transformation: {str(e)}")
            raise

    def explore_data(self, data_frames: Dict[str, pd.DataFrame] = None,
                     regions_dict: Dict = None) -> None:
        """
        Quick data exploration function for interactive analysis.
        
        Args:
            data_frames: Processed financial data. If None, uses self.processed_data.
            regions_dict: Regional classifications. If None, uses self.regions_dict.
        """
        if data_frames is None:
            data_frames = self.processed_data
        if regions_dict is None:
            regions_dict = self.regions_dict
        
        print("\n" + "=" * 50)
        print("QUICK DATA EXPLORATION")
        print("=" * 50)
        
        # Show available metrics
        print(f"\nAvailable Metrics ({len(data_frames)}):")
        metrics_list = list(data_frames.keys())
        for i, metric in enumerate(metrics_list[:15], 1):  # Show first 15
            print(f"   {i:2d}. {metric}")
        if len(metrics_list) > 15:
            print(f"   ... and {len(metrics_list) - 15} more metrics")
        
        # Show regional breakdowns
        print(f"\nRegional Classifications:")
        for region, countries in regions_dict.items():
            print(f"   {region:8s}: {len(countries):2d} countries")
        
        # Show sample data ranges
        if 'Price' in data_frames:
            price_data = data_frames['Price']
            print(f"\nSample Data Range (Price):")
            print(f"   Start Date: {price_data.index.min().strftime('%Y-%m-%d')}")
            print(f"   End Date:   {price_data.index.max().strftime('%Y-%m-%d')}")
            print(f"   Countries:  {len(price_data.columns)}")
        
        print(f"\nData ready for quantitative analysis!")

    def run_full_pipeline(self, export_data: bool = True) -> Tuple[Dict[str, pd.DataFrame], Dict, pd.DataFrame]:
        """
        Run the complete data processing pipeline.
        
        Args:
            export_data: Whether to export processed data to files (default: True).
        
        Returns:
            tuple: (processed_data, regions_dict, classification)
        """
        print("=" * 60)
        print("DATA PROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load raw data
        print("\nSTEP 1: Loading raw data from Excel files...")
        data_frames = self.read_excel_files_to_dict()
        file_names = list(data_frames.keys())
        self.original_file_names = file_names.copy()
        
        print(f"Successfully loaded {len(data_frames)} datasets:")
        for i, name in enumerate(file_names[:5], 1):
            print(f"   {i}. {name}")
        if len(file_names) > 5:
            print(f"   ... and {len(file_names) - 5} more datasets")
        
        if 'Assets' in data_frames:
            print(f"Sample data shape (Assets): {data_frames['Assets'].shape}")
        
        # Step 2: Load classification data
        print("\nSTEP 2: Loading classification data...")
        classification, classification_metricas, classification_map = self.load_classification_data()
        
        print(f"Classification data loaded:")
        print(f"   Countries: {len(classification)}")
        print(f"   Metrics mapping: {len(classification_map)} entries")
        
        # Step 3: Validate inputs
        print("\nSTEP 3: Validating input data...")
        self.validate_inputs()
        
        # Step 4: Remove weekends
        print("\nSTEP 4: Removing weekends from time series...")
        self.remove_weekends_optimized()
        print("Weekend removal completed for all datasets")
        
        # Step 5: Slice data and drop countries
        print("\nSTEP 5: Slicing data and filtering countries...")
        self.slice_data_frames_by_date()
        print("Data slicing and country filtering completed")
        
        # Step 6: Create regional classifications
        print("\nSTEP 6: Creating regional classifications...")
        regions_dict = self.get_regions_dict()
        
        print(f"Regional classifications created:")
        for region, countries in regions_dict.items():
            print(f"   {region}: {len(countries)} countries")
        
        # Step 7: Process and transform data
        print("\nSTEP 7: Processing and transforming financial data...")
        print("This includes:")
        print("   • Regional aggregations")
        print("   • Growth metrics calculation")
        print("   • Yield computations")
        print("   • Spread analysis")
        print("   • Rolling statistics")
        print("   • Balance sheet ratios")
        
        processed_data = self.transform_process_data(export_data=export_data)
        
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\nPROCESSING SUMMARY:")
        print(f"   • Original datasets: {len(file_names)}")
        print(f"   • Final processed datasets: {len(processed_data)}")
        print(f"   • Countries in classification: {len(classification)}")
        print(f"   • Regional groupings: {len(regions_dict)}")
        
        if export_data:
            print(f"   • Export folder: outputs/processed_inputs/")
        
        derived_metrics = [k for k in processed_data.keys() if k not in file_names]
        if derived_metrics:
            print(f"\nNEW DERIVED METRICS ({len(derived_metrics)}):")
            for i, metric in enumerate(derived_metrics[:10], 1):
                print(f"   {i}. {metric}")
            if len(derived_metrics) > 10:
                print(f"   ... and {len(derived_metrics) - 10} more derived metrics")
        
        return processed_data, regions_dict, classification

