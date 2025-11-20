"""
IC Analysis for Normalized Scores

Calculates Information Coefficient statistics for different normalized score methods
across multiple periodicities and methodologies (Absolute vs Relative).
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load prices and normalized scores."""
    print("Loading data...")
    
    # Load prices
    prices = pd.read_excel('ProcessedInputs/Price.xlsx', index_col=0, parse_dates=True)
    print(f"✓ Prices loaded: {prices.shape}")
    
    # Clean prices: keep only numeric columns
    print(f"  Cleaning price data...")
    original_cols = prices.columns.tolist()
    
    # Select only numeric columns
    numeric_cols = prices.select_dtypes(include=[np.number]).columns.tolist()
    
    # Also try to convert columns that might be stored as strings
    for col in prices.columns:
        if col not in numeric_cols:
            try:
                prices[col] = pd.to_numeric(prices[col], errors='coerce')
                if prices[col].notna().sum() > 0:  # If conversion worked for some values
                    numeric_cols.append(col)
            except:
                pass
    
    # Keep only numeric columns
    prices = prices[numeric_cols]
    
    dropped_cols = set(original_cols) - set(numeric_cols)
    if dropped_cols:
        print(f"  ⚠ Dropped non-numeric columns: {dropped_cols}")
    
    print(f"  ✓ Price data cleaned: {prices.shape} (numeric columns only)")
    print(f"  Columns: {list(prices.columns[:5])}{'...' if len(prices.columns) > 5 else ''}")
    
    # Load all normalized scores
    normalized_scores_dict = {}
    score_files = glob.glob('Normalized_Scores/*.xlsx')
    
    for file_path in sorted(score_files):
        file_name = Path(file_path).stem
        score_df = pd.read_excel(file_path, index_col=0, parse_dates=True)
        
        # Clean normalized scores: keep only numeric columns
        numeric_cols_score = score_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Try to convert string columns to numeric
        for col in score_df.columns:
            if col not in numeric_cols_score:
                try:
                    score_df[col] = pd.to_numeric(score_df[col], errors='coerce')
                    if score_df[col].notna().sum() > 0:
                        numeric_cols_score.append(col)
                except:
                    pass
        
        score_df = score_df[numeric_cols_score]
        normalized_scores_dict[file_name] = score_df
        
    print(f"✓ Loaded {len(normalized_scores_dict)} normalized scores")
    
    # Final validation
    if prices.empty:
        raise ValueError("ERROR: Prices DataFrame is empty after cleaning!")
    
    if len(normalized_scores_dict) == 0:
        raise ValueError("ERROR: No normalized scores loaded!")
    
    # Check for any remaining non-numeric data
    non_numeric_prices = prices.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_prices:
        raise ValueError(f"ERROR: Prices still contains non-numeric columns: {non_numeric_prices}")
    
    print(f"✓ Data validation passed\n")
    
    return prices, normalized_scores_dict


def calculate_forward_returns(prices, periodicity):
    """
    Calculate forward returns for given periodicity.
    
    Returns[t] = (Price[t+period] - Price[t]) / Price[t]
    """
    try:
        # Ensure all data is numeric
        prices_numeric = prices.apply(pd.to_numeric, errors='coerce')
        
        # Calculate returns: (price[t+period] - price[t]) / price[t]
        forward_returns = prices_numeric.shift(-periodicity) / prices_numeric - 1
        
        return forward_returns
    except Exception as e:
        print(f"\n✗ ERROR calculating forward returns:")
        print(f"  Periodicity: {periodicity}")
        print(f"  Prices shape: {prices.shape}")
        print(f"  Prices dtypes:\n{prices.dtypes}")
        raise


def calculate_signal(normalized_score, periodicity, method):
    """
    Calculate signal based on method.
    
    For ABSOLUTE method:
        Signal = current normalized score
        Tests if high scores predict high forward returns
        
    For RELATIVE method:
        Signal = score change over periodicity
        Tests if score improvements predict high forward returns
        Example: periodicity=21 → signal = score[t] - score[t-21]
    
    Parameters
    ----------
    normalized_score : pd.DataFrame
        Normalized scores
    periodicity : int
        Period for signal calculation
        - For absolute: not used (current scores)
        - For relative: lookback period for score change
    method : str
        'absolute' or 'relative'
    
    Returns
    -------
    pd.DataFrame
        Signal values
    """
    if method == 'absolute':
        # ABSOLUTE: Use current scores as signal
        # Signal[t] = Score[t]
        return normalized_score
    else:  # relative
        # RELATIVE: Use score changes over periodicity as signal
        # Signal[t] = Score[t] - Score[t-periodicity]
        # This tests if score momentum predicts returns
        return normalized_score.diff(periodicity)


def calculate_ic_statistics(signal_df, forward_returns_df, periodicity):
    """
    Calculate IC statistics between signal and forward returns.
    
    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal values (scores or score changes)
    forward_returns_df : pd.DataFrame
        Forward returns
    periodicity : int
        Period used
        
    Returns
    -------
    dict
        IC statistics
    """
    ic_values = []
    
    # Align dates
    common_dates = signal_df.index.intersection(forward_returns_df.index)
    
    for date in common_dates:
        # Get signal and forward returns at this date
        signal = signal_df.loc[date]
        fwd_returns = forward_returns_df.loc[date]
        
        # Get common countries (both have data)
        common_countries = signal.dropna().index.intersection(fwd_returns.dropna().index)
        
        if len(common_countries) >= 3:  # Need at least 3 points
            signal_values = signal[common_countries].values
            return_values = fwd_returns[common_countries].values
            
            # Calculate Spearman rank correlation
            ic, p_value = spearmanr(signal_values, return_values)
            ic_values.append(ic)
    
    if len(ic_values) == 0:
        return {
            'mean_ic': np.nan,
            'median_ic': np.nan,
            'std_ic': np.nan,
            'ic_t_stat': np.nan,
            'icir': np.nan,
            'hit_rate': np.nan,
            'num_periods': 0
        }
    
    ic_series = pd.Series(ic_values)
    
    # Calculate statistics
    mean_ic = ic_series.mean()
    median_ic = ic_series.median()
    std_ic = ic_series.std()
    n_periods = len(ic_series)
    
    # IC t-statistic
    ic_t_stat = mean_ic / (std_ic / np.sqrt(n_periods)) if std_ic != 0 else 0
    
    # ICIR (IC Information Ratio)
    icir = mean_ic / std_ic if std_ic != 0 else 0
    
    # Hit rate
    hit_rate = (ic_series > 0).sum() / len(ic_series)
    
    return {
        'mean_ic': mean_ic,
        'median_ic': median_ic,
        'std_ic': std_ic,
        'ic_t_stat': ic_t_stat,
        'icir': icir,
        'hit_rate': hit_rate,
        'num_periods': n_periods
    }


def run_ic_analysis(prices, normalized_scores_dict, periodicities=[5, 10, 21, 63], 
                    methods=['absolute', 'relative']):
    """
    Run IC analysis for all combinations.
    
    For each combination of score × periodicity × method, calculates:
    
    ABSOLUTE METHOD:
        Signal: Current normalized score
        Forward Returns: Price returns over periodicity
        IC: Correlation(Score[t], Return[t to t+period])
        
    RELATIVE METHOD:  
        Signal: Score change over periodicity (Score[t] - Score[t-period])
        Forward Returns: Price returns over periodicity
        IC: Correlation(Score_Change[t-period to t], Return[t to t+period])
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    normalized_scores_dict : dict
        Dictionary of normalized score DataFrames
    periodicities : list
        List of periodicities to test (default [5, 10, 21, 63])
    methods : list
        List of methods to test ('absolute', 'relative')
        
    Returns
    -------
    pd.DataFrame
        Results DataFrame with IC statistics
    """
    results = []
    
    total_tests = len(normalized_scores_dict) * len(periodicities) * len(methods)
    counter = 0
    
    print(f"Running IC Analysis: {total_tests} tests")
    print(f"  - {len(normalized_scores_dict)} normalized scores")
    print(f"  - {len(periodicities)} periodicities: {periodicities}")
    print(f"  - {len(methods)} methods: {methods}")
    print("\nTest Matrix:")
    print("  Method    | Periodicity | Signal Definition              | Forward Returns")
    print("  " + "-" * 75)
    for method in methods:
        for period in periodicities:
            if method == 'absolute':
                signal_def = f"Score[t]"
                fwd_ret = f"Return[t → t+{period}]"
            else:
                signal_def = f"Score[t] - Score[t-{period}]"
                fwd_ret = f"Return[t → t+{period}]"
            print(f"  {method:<9} | {period:>11} | {signal_def:<30} | {fwd_ret}")
    print("=" * 80)
    
    for score_name, normalized_score in normalized_scores_dict.items():
        for periodicity in periodicities:
            try:
                # Calculate forward returns
                forward_returns = calculate_forward_returns(prices, periodicity)
            except Exception as e:
                print(f"\n✗ ERROR calculating forward returns for periodicity {periodicity}")
                print(f"   {str(e)}")
                continue
            
            for method in methods:
                counter += 1
                
                try:
                    # Calculate signal
                    signal = calculate_signal(normalized_score, periodicity, method)
                    
                    # Calculate IC statistics
                    ic_stats = calculate_ic_statistics(signal, forward_returns, periodicity)
                    
                    # Store results
                    results.append({
                        'Score_Name': score_name,
                        'Periodicity': periodicity,
                        'Method': method,
                        'Mean_IC': ic_stats['mean_ic'],
                        'Median_IC': ic_stats['median_ic'],
                        'Std_IC': ic_stats['std_ic'],
                        'IC_t_stat': ic_stats['ic_t_stat'],
                        'ICIR': ic_stats['icir'],
                        'Hit_Rate': ic_stats['hit_rate'],
                        'Num_Periods': ic_stats['num_periods']
                    })
                except Exception as e:
                    print(f"\n✗ ERROR in test {counter}: {score_name} | P={periodicity} | {method}")
                    print(f"   {str(e)}")
                    # Store error result
                    results.append({
                        'Score_Name': score_name,
                        'Periodicity': periodicity,
                        'Method': method,
                        'Mean_IC': np.nan,
                        'Median_IC': np.nan,
                        'Std_IC': np.nan,
                        'IC_t_stat': np.nan,
                        'ICIR': np.nan,
                        'Hit_Rate': np.nan,
                        'Num_Periods': 0
                    })
                
                # Progress update
                if counter % 10 == 0 or counter == total_tests:
                    print(f"Progress: {counter}/{total_tests} tests completed")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("IC Analysis Complete!")
    print("=" * 80)
    
    return results_df


def print_results_summary(results_df):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("TOP PERFORMERS BY MEAN IC")
    print("=" * 80)
    
    # Overall top 10
    print("\nTop 10 Overall:")
    top_10 = results_df.nlargest(10, 'Mean_IC')[['Score_Name', 'Periodicity', 'Method', 'Mean_IC', 'IC_t_stat', 'ICIR']]
    print(top_10.to_string(index=False))
    
    # By periodicity
    print("\n" + "=" * 80)
    print("TOP PERFORMERS BY PERIODICITY")
    print("=" * 80)
    for period in sorted(results_df['Periodicity'].unique()):
        print(f"\nPeriodicity = {period} days:")
        print(f"  (Absolute: score → {period}-day forward return)")
        print(f"  (Relative: {period}-day score change → {period}-day forward return)")
        subset = results_df[results_df['Periodicity'] == period].nlargest(5, 'Mean_IC')
        for _, row in subset.iterrows():
            print(f"  {row['Score_Name'][:60]:<60} | {row['Method']:<8} | IC: {row['Mean_IC']:>7.4f} | t: {row['IC_t_stat']:>6.2f}")
    
    # By method
    print("\n" + "=" * 80)
    print("TOP PERFORMERS BY METHOD")
    print("=" * 80)
    for method in ['absolute', 'relative']:
        print(f"\nMethod = {method}:")
        subset = results_df[results_df['Method'] == method].nlargest(5, 'Mean_IC')
        for _, row in subset.iterrows():
            print(f"  {row['Score_Name'][:60]:<60} | P={row['Periodicity']:<2} | IC: {row['Mean_IC']:>7.4f} | t: {row['IC_t_stat']:>6.2f}")


def export_results(results_df, output_file='ic_analysis_results.csv'):
    """Export results to CSV."""
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results exported to: {output_file}")


def main():
    """Main execution function."""
    # Load data
    prices, normalized_scores_dict = load_data()
    
    # Run IC analysis
    results_df = run_ic_analysis(
        prices=prices,
        normalized_scores_dict=normalized_scores_dict,
        periodicities=[5, 10, 21, 63],
        methods=['absolute', 'relative']
    )
    
    # Print summary
    print_results_summary(results_df)
    
    # Export results
    export_results(results_df)
    
    # Export to Excel with better formatting
    with pd.ExcelWriter('ic_analysis_results.xlsx', engine='openpyxl') as writer:
        # Full results
        results_df.to_excel(writer, sheet_name='Full_Results', index=False)
        
        # Pivot tables for easier comparison
        for method in ['absolute', 'relative']:
            pivot = results_df[results_df['Method'] == method].pivot_table(
                values='Mean_IC',
                index='Score_Name',
                columns='Periodicity'
            )
            pivot.to_excel(writer, sheet_name=f'{method.capitalize()}_IC')
        
        # Top performers
        top_20 = results_df.nlargest(20, 'Mean_IC')
        top_20.to_excel(writer, sheet_name='Top_20', index=False)
    
    print(f"✓ Results exported to: ic_analysis_results.xlsx")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
