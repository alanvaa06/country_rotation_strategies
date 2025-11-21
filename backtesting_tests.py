"""
Backtesting Tests Module

This module provides a systematic framework for running multiple backtests
with different parameter configurations and storing performance results.

Features:
- Automated data loading from ProcessData
- Support for multiple normalized score scenarios
- Parameter grid testing (selection criteria, thresholds, weighting methods, etc.)
- Structured results storage (DataFrame format)
- Export results to Excel for analysis

Author: Alan Vazquez, CFA, Master Jedi
Last Updated: November 2025
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import warnings
import re
from datetime import datetime
from itertools import product
import pandas as pd
import numpy as np

# Local imports
from ProcessData import ProcessData
from backtest import Backtest

# Configuration
warnings.filterwarnings('ignore')

# ============================================================================
# BACKTESTING TEST CLASS
# ============================================================================

class BacktestingTests:
    """
    A class to manage and run multiple backtests systematically.
    
    This class handles:
    - Data loading and preparation
    - Parameter grid generation
    - Backtest execution
    - Results aggregation and storage
    
    Parameters
    ----------
    inputs_folder : str, optional
        Path to inputs folder (default: 'Inputs')
    classification_file : str, optional
        Classification file name (default: 'Classification.xlsx')
    target_date : str, optional
        Start date for data processing (default: '2010-01-01')
    columns_to_drop : list, optional
        Countries to exclude (default: ['Saudi Arabia'])
    normalized_scores_folder : str, optional
        Folder containing normalized scores (default: 'Normalized_Scores')
    results_folder : str, optional
        Folder to save results (default: 'Backtest_Results')
    """
    
    def __init__(
        self,
        inputs_folder='Inputs',
        classification_file='Classification.xlsx',
        target_date='2010-01-01',
        columns_to_drop=['Saudi Arabia'],
        normalized_scores_folder='Normalized_Scores',
        results_folder='Backtest_Results'
    ):
        """Initialize the BacktestingTests class."""
        self.inputs_folder = inputs_folder
        self.classification_file = classification_file
        self.target_date = target_date
        self.columns_to_drop = columns_to_drop
        self.normalized_scores_folder = normalized_scores_folder
        self.results_folder = results_folder
        
        # Data containers
        self.prices = None
        self.normalized_scores = None
        self.results_df = None
        
        # Create results folder if it doesn't exist
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
            # print(f"Created results folder: {self.results_folder}")
    
    def load_price_data(self):
        """
        Load and process price data using ProcessData.
        
        Returns
        -------
        pd.DataFrame
            Price data with dates as index and countries as columns
        """
        processor = ProcessData(
            inputs_folder=self.inputs_folder,
            classification_file=self.classification_file,
            target_date=self.target_date,
            columns_to_drop=self.columns_to_drop
        )
        
        processed_data, regions_dict, classification = processor.run_full_pipeline(export_data=False)
        self.prices = processed_data['Price']
        
        # Note: Benchmark is not set here anymore as it depends on the score file.
        
        return self.prices
    
    def load_normalized_scores(self, filename):
        """
        Load normalized scores from file.
        
        Parameters
        ----------
        filename : str
            Name of the normalized scores file (e.g., 'NormalizedScores_EM_...xlsx')
        
        Returns
        -------
        pd.DataFrame
            Normalized scores with dates as index and countries as columns
        """
        file_path = os.path.join(self.normalized_scores_folder, filename)
        
        try:
            self.normalized_scores = pd.read_excel(
                file_path,
                index_col=0,
                parse_dates=True
            )
            
            return self.normalized_scores
            
        except FileNotFoundError:
            print(f"⚠ ERROR: File not found: {file_path}")
            raise
        except Exception as e:
            print(f"⚠ ERROR loading normalized scores: {str(e)}")
            raise
    
    def get_normalized_score_files(self, pattern=None):
        """
        Get list of normalized score files in the folder.
        
        Parameters
        ----------
        pattern : str, optional
            Filter files by pattern (e.g., 'EM_', 'Asia_', etc.)
        
        Returns
        -------
        list
            List of normalized score filenames
        """
        all_files = [f for f in os.listdir(self.normalized_scores_folder) 
                     if f.endswith('.xlsx') and f.startswith('NormalizedScores_')]
        
        if pattern:
            all_files = [f for f in all_files if pattern in f]
        
        return sorted(all_files)
    
    def extract_benchmark_from_filename(self, filename):
        """
        Extract benchmark name from the normalized score filename.
        
        Logic: Benchmark is the asset just after 'NormalizedScores_' and before '_Scenario'.
        Example: 'NormalizedScores_Asia_Scenario_1...' -> 'Asia'
        
        Parameters
        ----------
        filename : str
            Normalized score filename
            
        Returns
        -------
        str
            Extracted benchmark name
        """
        match = re.search(r"NormalizedScores_(.+?)_Scenario", filename)
        if match:
            return match.group(1)
        else:
            print(f"⚠ Warning: Could not extract benchmark from filename '{filename}'. Defaulting to 'World'.")
            return 'World'
    
    def create_parameter_grid(self, params_dict):
        """
        Create parameter grid for backtesting.
        
        Parameters
        ----------
        params_dict : dict
            Dictionary where keys are parameter names and values are lists of values to test
            Example:
            {
                'selection_criteria': ['absolute', 'relative'],
                'absolute_selection_score': [0.60, 0.70, 0.75, 0.80],
                'weighting_method': ['Equal', 'Risk_Parity'],
                'bmk_weight': [0.0, 0.3, 0.5]
            }
        
        Returns
        -------
        list of dict
            List of parameter dictionaries for each configuration
        """
        # Get all parameter names and their values
        param_names = list(params_dict.keys())
        param_values = [params_dict[key] for key in param_names]
        
        # Create all combinations
        combinations = list(product(*param_values))
        
        # Convert to list of dictionaries
        param_grid = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_grid.append(param_dict)
        
        return param_grid
    
    def run_single_backtest(self, params, test_name=None, benchmark=None):
        """
        Run a single backtest with given parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary of backtest parameters
        test_name : str, optional
            Name for this test configuration
        benchmark : str, optional
            Name of the benchmark column in prices. If provided, overrides params['bmk'].
        
        Returns
        -------
        dict
            Dictionary containing test configuration and performance metrics
        """
        # Update params with benchmark if provided
        if benchmark:
            params = params.copy()
            params['bmk'] = benchmark
            
        # Create backtest instance
        bt = Backtest(
            normalized_score=self.normalized_scores,
            prices=self.prices,
            **params
        )
        
        # Run backtest
        bt.run_backtest()
        
        # Get performance summary
        # Transpose to get metrics as index and Entities (Portfolio, Benchmark, Active) as columns
        summary = bt.get_performance_summary().T
        
        # Extract key metrics
        result = {
            'test_name': test_name if test_name else 'Unnamed',
            'benchmark_asset': params.get('bmk', 'Unknown'),
            'selection_criteria': params.get('selection_criteria', 'absolute'),
            'absolute_selection_score': params.get('absolute_selection_score', None),
            'relative_selection_score': params.get('relative_selection_score', None),
            'weighting_method': params.get('weighting_method', 'Equal'),
            'bmk_weight': params.get('bmk_weight', 0.0),
            'periodicity': params.get('periodicity', 5),
            'transaction_cost_bps': params.get('transaction_cost_bps', 2.0),
            'risk_parity_lookback': params.get('risk_parity_lookback', 60),
        }
        
        # Add metrics
        for metric in summary.index:
            if 'Portfolio' in summary.columns:
                result[f'Portfolio_{metric}'] = summary.loc[metric, 'Portfolio']
            if 'Benchmark' in summary.columns:
                result[f'Benchmark_{metric}'] = summary.loc[metric, 'Benchmark']
            if 'Active' in summary.columns:
                result[f'Active_{metric}'] = summary.loc[metric, 'Active']
        
        return result
    
    def run_parameter_grid_tests(self, param_grid, test_prefix="Test"):
        """
        Run backtests for all parameter combinations in the grid.
        
        Parameters
        ----------
        param_grid : list of dict
            List of parameter dictionaries
        test_prefix : str, optional
            Prefix for test names (default: 'Test')
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all backtest results
        """
        print("\n" + "=" * 80)
        print(f"RUNNING PARAMETER GRID TESTS ({len(param_grid)} configurations)")
        print("=" * 80)
        
        results = []
        
        for idx, params in enumerate(param_grid, 1):
            test_name = f"{test_prefix}_{idx}"
            
            # Minimal printing or progress
            if idx % 10 == 0 or idx == 1 or idx == len(param_grid):
                print(f"[{idx}/{len(param_grid)}] Running {test_name}...")
            
            try:
                result = self.run_single_backtest(params, test_name)
                results.append(result)
                
            except Exception as e:
                print(f"   ⚠ Error in {test_name}: {str(e)}")
                continue
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(results)
        
        print("\n" + "=" * 80)
        print(f"COMPLETED: {len(results)} successful backtests")
        print("=" * 80)
        
        return self.results_df
    
    def run_multi_score_tests(self, score_files, param_grid, test_prefix="MultiScore"):
        """
        Run backtests across multiple normalized score files.
        
        Parameters
        ----------
        score_files : list
            List of normalized score filenames
        param_grid : list of dict
            List of parameter dictionaries (configurations) to test
        test_prefix : str, optional
            Prefix for test names
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all backtest results across score files
        """
        print("\n" + "=" * 80)
        print(f"RUNNING MULTI-SCORE TESTS")
        print(f"   Score files: {len(score_files)}")
        print(f"   Configurations per file: {len(param_grid)}")
        print("=" * 80)
        
        all_results = []
        
        for score_idx, score_file in enumerate(score_files, 1):
            print(f"\nProcessing file {score_idx}/{len(score_files)}: {score_file}")
            
            # Extract benchmark from filename
            benchmark = self.extract_benchmark_from_filename(score_file)
            print(f"   • Detected Benchmark: {benchmark}")
            
            # Load normalized scores
            try:
                self.load_normalized_scores(score_file)
            except Exception as e:
                print(f"⚠ Skipping {score_file}: {str(e)}")
                continue
            
            # Run tests for this score file
            for idx, params in enumerate(param_grid, 1):
                test_name = f"{test_prefix}_{score_idx}_{idx}"
                
                if idx % 50 == 0:  # Print less frequently
                     print(f"   [{idx}/{len(param_grid)}] Running {test_name}...")
                
                try:
                    # Pass benchmark explicitly
                    result = self.run_single_backtest(params, test_name, benchmark=benchmark)
                    result['score_file'] = score_file
                    result['score_index'] = score_idx
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"   ⚠ Error in {test_name}: {str(e)}")
                    continue
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(all_results)
        
        print("\n" + "=" * 80)
        print(f"COMPLETED: {len(all_results)} successful backtests")
        print("=" * 80)
        
        return self.results_df
    
    def save_results(self, filename=None):
        """
        Save results to Excel file.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, generates timestamp-based name
        
        Returns
        -------
        str
            Path to saved file
        """
        if self.results_df is None or len(self.results_df) == 0:
            print("⚠ No results to save")
            return None
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.xlsx"
        
        # Ensure .xlsx extension
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'
        
        # Full path
        filepath = os.path.join(self.results_folder, filename)
        
        # Save to Excel
        self.results_df.to_excel(filepath, index=False)
        
        print(f"\n✓ Results saved to: {filepath}")
        print(f"   • Total tests: {len(self.results_df)}")
        print(f"   • Columns: {len(self.results_df.columns)}")
        
        return filepath
    
    def get_top_performers(self, metric='Portfolio_Sharpe Ratio', n=10):
        """
        Get top N performing configurations by specified metric.
        
        Parameters
        ----------
        metric : str, optional
            Metric to sort by (default: 'Portfolio_Sharpe Ratio')
        n : int, optional
            Number of top performers to return (default: 10)
        
        Returns
        -------
        pd.DataFrame
            Top N performing configurations
        """
        if self.results_df is None or len(self.results_df) == 0:
            print("⚠ No results available")
            return None
        
        # Check if metric exists
        if metric not in self.results_df.columns:
            print(f"⚠ Metric '{metric}' not found in results. Available metrics: {list(self.results_df.columns)}")
            return None
        
        top = self.results_df.nlargest(n, metric)
        
        return top
    
    def print_summary_statistics(self):
        """Print summary statistics of all backtest results."""
        if self.results_df is None or len(self.results_df) == 0:
            print("⚠ No results available")
            return
        
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        # Key metrics to summarize
        metrics = [
            'Portfolio_Annualized Return',
            'Portfolio_Annualized Volatility',
            'Portfolio_Sharpe Ratio',
            'Portfolio_Max Drawdown',
            'Active_Annualized Return',
            'Portfolio_Win Rate'
        ]
        
        print(f"\nTotal Tests: {len(self.results_df)}")
        print("\nMetric Statistics:")
        print("-" * 80)
        
        for metric in metrics:
            if metric in self.results_df.columns:
                mean_val = self.results_df[metric].mean()
                std_val = self.results_df[metric].std()
                min_val = self.results_df[metric].min()
                max_val = self.results_df[metric].max()
                
                print(f"\n{metric}:")
                print(f"   Mean:   {mean_val:>8.4f}")
                print(f"   Std:    {std_val:>8.4f}")
                print(f"   Min:    {min_val:>8.4f}")
                print(f"   Max:    {max_val:>8.4f}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("BACKTESTING TESTS - SYSTEMATIC PARAMETER TESTING")
    print("=" * 80)
    
    # Initialize testing framework
    bt_tests = BacktestingTests()
    
    # Load price data
    bt_tests.load_price_data()
    
    # ========================================================================
    # SYSTEMATIC TESTING: All Normalized Scores + Parameter Grid
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SYSTEMATIC TESTING: FILTERED SCORES & PARAMETERS")
    print("=" * 80)
    
    # Load IC Analysis Results to filter score files
    ic_analysis_file = "ic_analysis_results.xlsx"
    ic_threshold = 0.05
    
    print(f"\nLoading IC Analysis from: {ic_analysis_file}")
    print(f"Filtering for Mean_IC > {ic_threshold}")
    
    try:
        if os.path.exists(ic_analysis_file):
            ic_df = pd.read_excel(ic_analysis_file)
            
            # Filter by Mean_IC
            filtered_scores = ic_df[ic_df['Mean_IC'] > ic_threshold]
            
            # Get file names (ensure .xlsx extension)
            score_files = filtered_scores['Score_Name'].tolist()
            # Add extension if missing
            score_files = [f if str(f).endswith('.xlsx') else f"{f}.xlsx" for f in score_files]
            
            print(f"Found {len(score_files)} score files meeting criteria (out of {len(ic_df)})")
        else:
            print(f"⚠ File not found: {ic_analysis_file}")
            print("Falling back to all files in folder...")
            score_files = bt_tests.get_normalized_score_files()

    except Exception as e:
        print(f"⚠ Error loading or filtering IC analysis file: {str(e)}")
        print("Falling back to all files in folder...")
        score_files = bt_tests.get_normalized_score_files()
    
    print(f"\nTotal files to process: {len(score_files)}")
    
    # Define parameter grids
    # Common parameters
    common_params = {
        'weighting_method': ['Equal', 'Risk_Parity'],
        'bmk_weight': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'periodicity': [5, 10, 21, 63],
        'transaction_cost_bps': [2.0],
        'risk_parity_lookback': [120]
    }

    # Grid 1: Absolute Selection
    # Varies absolute_selection_score
    params_absolute = common_params.copy()
    params_absolute['selection_criteria'] = ['absolute']
    params_absolute['absolute_selection_score'] = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    # Grid 2: Relative Selection
    # Does NOT vary absolute_selection_score (uses default or ignored)
    # We assume relative_selection_score uses default (5) as user didn't specify list
    params_relative = common_params.copy()
    params_relative['selection_criteria'] = ['relative']
    
    # Create grids
    grid_abs = bt_tests.create_parameter_grid(params_absolute)
    grid_rel = bt_tests.create_parameter_grid(params_relative)
    
    # Combine grids
    full_grid = grid_abs + grid_rel
    
    print(f"Total configurations per file: {len(full_grid)}")
    print(f"  - Absolute: {len(grid_abs)}")
    print(f"  - Relative: {len(grid_rel)}")
    
    # Run tests across all score files
    # Using a prefix to identify these tests
    results = bt_tests.run_multi_score_tests(
        score_files=score_files,
        param_grid=full_grid,
        test_prefix="SysTest"
    )
    
    # Save results
    saved_file = bt_tests.save_results("backtest_results_comprehensive.xlsx")
    
    # Print summary
    bt_tests.print_summary_statistics()
    
    # Show top performers based on different metrics
    if results is not None and not results.empty:
        print("\n" + "=" * 80)
        print("TOP 5 PERFORMERS (by Active Information Ratio)")
        print("=" * 80)
        if 'Active_Information Ratio' in results.columns:
            top_ir = bt_tests.get_top_performers(metric='Active_Information Ratio', n=5)
            cols_to_show = ['score_file', 'benchmark_asset', 'selection_criteria', 
                           'absolute_selection_score', 'weighting_method', 'bmk_weight', 
                           'periodicity', 'Portfolio_Annualized Return', 
                           'Active_Annualized Return', 'Active_Information Ratio']
            # Filter cols that exist
            cols_to_show = [c for c in cols_to_show if c in top_ir.columns]
            print("\n", top_ir[cols_to_show].to_string())
        else:
            print("Metric 'Active_Information Ratio' not found.")

    print("\n" + "=" * 80)
    print("BACKTESTING TESTS COMPLETED!")
    print("=" * 80)
