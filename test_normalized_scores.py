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

#%%
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


def calculate_period_returns(prices, periodicity):
    """
    Calculate returns between consecutive rebalancing dates.
    
    Samples prices at fixed intervals (every 'periodicity' days) and calculates
    the return between consecutive rebalancing dates:
    
        Returns[t1] = (Price[t1] - Price[t0]) / Price[t0]
    
    where t0 and t1 are consecutive selected dates.
    
    Note: The "forward" relationship (signal → return) is established in the 
    IC calculation where signal[t0] is paired with return[t0→t1].
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data with datetime index
    periodicity : int
        Interval between rebalancing dates (in days)
        
    Returns
    -------
    pd.DataFrame
        Period returns between consecutive rebalancing dates
    """

    all_dates = prices.index
    selected_dates = sorted(all_dates[::-periodicity])
    
    try:
        # Ensure all data is numeric
        prices_numeric = prices.apply(pd.to_numeric, errors='coerce')
        
        # Calculate returns between consecutive rebalancing dates
        period_returns = prices_numeric.loc[selected_dates].pct_change() 
        period_returns = period_returns.iloc[1:]
        return period_returns

    except Exception as e:
        print(f"\n✗ ERROR calculating period returns:")
        print(f"  Periodicity: {periodicity}")
        print(f"  Prices shape: {prices.shape}")
        print(f"  Prices dtypes:\n{prices.dtypes}")
        raise


def calculate_signal(normalized_score, periodicity, method):
    """
    Calculate signal at rebalancing dates based on method.
    
    Samples dates at fixed intervals (every 'periodicity' days) and calculates
    signals that will be tested for predictive power.
    
    ABSOLUTE METHOD:
        Signal[t] = Score[t]
        Uses the score level at each rebalancing date
        Tests if high scores predict high subsequent returns
        
    RELATIVE METHOD:
        Signal[t] = Score[t] - Score[t-period]
        Uses the score change between consecutive rebalancing dates
        Tests if score momentum predicts high subsequent returns
    
    Parameters
    ----------
    normalized_score : pd.DataFrame
        Normalized scores with datetime index
    periodicity : int
        Interval between rebalancing dates (in days)
        - For absolute: defines sampling frequency
        - For relative: defines both sampling and differencing period
    method : str
        'absolute' or 'relative'
    
    Returns
    -------
    pd.DataFrame
        Signal values at rebalancing dates
    """

    all_dates = normalized_score.index
    selected_dates = sorted(all_dates[::-periodicity])

    if method == 'absolute':
        # ABSOLUTE: Use current scores as signal
        # Signal[t] = Score[t]
        return normalized_score.loc[selected_dates]
    else:  # relative
        # RELATIVE: Use score changes over periodicity as signal
        # Signal[t] = Score[t] - Score[t-periodicity]
        # This tests if score momentum predicts returns
        return normalized_score.loc[selected_dates].diff().iloc[1:]


def calculate_ic_statistics(signal_df, period_returns_df, periodicity):
    """
    Calculate IC statistics between signal and period returns.
    
    This function establishes the forward-looking predictive relationship by pairing:
        - Signal at period start (t0) 
        - Return during the period (t0 → t1)
    
    The IC measures: Correlation(Signal[t0], Return[t0→t1])
    
    This tests whether the signal at the start of a period predicts returns 
    during that period, which is the correct approach for a rebalancing strategy.
    
    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal values at rebalancing dates (scores or score changes)
    period_returns_df : pd.DataFrame
        Period returns between consecutive rebalancing dates
    periodicity : int
        Period interval (for reference)
        
    Returns
    -------
    dict
        IC statistics including mean_ic, median_ic, std_ic, ic_t_stat, 
        icir, hit_rate, and num_periods
    """
    ic_values = []
    
    # Align dates and create (start, end) pairs for each period
    common_dates = signal_df.index.intersection(period_returns_df.index)
    dates=[(common_dates[i], common_dates[i+1]) for i in range(len(common_dates)-1)]
    
    for date in dates:
        # Get signal at period start and return at period end (which represents t0→t1 return)
        signal = signal_df.loc[date[0]]
        period_returns = period_returns_df.loc[date[1]]
        
        # Get common countries (both have data)
        common_countries = signal.dropna().index.intersection(period_returns.dropna().index)
        
        if len(common_countries) >= 3:  # Need at least 3 points
            signal_values = signal[common_countries].values
            return_values = period_returns[common_countries].values
                
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
    hit_rate = (ic_series > 0).sum() / len(ic_series)  if len(ic_series) > 0 else 0
    
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
    Run IC analysis for all combinations of score × periodicity × method.
    
    For each combination:
    1. Samples dates at fixed intervals (every 'periodicity' days)
    2. Calculates period returns between consecutive rebalancing dates
    3. Calculates signals at rebalancing dates
    4. Computes IC: Correlation(Signal[t0], Return[t0→t1])
    
    ABSOLUTE METHOD:
        Signal[t0]: Normalized score at period start
        Return[t0→t1]: Price return during period
        IC: Tests if high scores predict high subsequent returns
        
    RELATIVE METHOD:  
        Signal[t0]: Score change (Score[t0] - Score[t0-period])
        Return[t0→t1]: Price return during period
        IC: Tests if score momentum predicts high subsequent returns
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data with datetime index
    normalized_scores_dict : dict
        Dictionary of normalized score DataFrames
    periodicities : list
        List of rebalancing intervals to test (default [5, 10, 21, 63])
    methods : list
        List of methods to test ('absolute', 'relative')
        
    Returns
    -------
    pd.DataFrame
        Results DataFrame with IC statistics for all combinations
    """
    results = []
    
    total_tests = len(normalized_scores_dict) * len(periodicities) * len(methods)
    counter = 0
    
    print(f"Running IC Analysis: {total_tests} tests")
    print(f"  - {len(normalized_scores_dict)} normalized scores")
    print(f"  - {len(periodicities)} periodicities: {periodicities}")
    print(f"  - {len(methods)} methods: {methods}")
    print("\nTest Matrix:")
    print("  Method    | Periodicity | Signal Definition              | Period Returns")
    print("  " + "-" * 75)
    for method in methods:
        for period in periodicities:
            if method == 'absolute':
                signal_def = f"Score[t]"
                period_ret = f"Return[t → t+{period}]"
            else:
                signal_def = f"Score[t] - Score[t-{period}]"
                period_ret = f"Return[t → t+{period}]"
            print(f"  {method:<9} | {period:>11} | {signal_def:<30} | {period_ret}")
    print("=" * 80)
    
    for score_name, normalized_score in normalized_scores_dict.items():
        for periodicity in periodicities:
            try:
                # Calculate period returns between rebalancing dates
                period_returns = calculate_period_returns(prices, periodicity)
            except Exception as e:
                print(f"\n✗ ERROR calculating period returns for periodicity {periodicity}")
                print(f"   {str(e)}")
                continue
            
            for method in methods:
                counter += 1
                
                try:
                    # Calculate signal
                    signal = calculate_signal(normalized_score, periodicity, method)
                    
                    # Calculate IC statistics
                    ic_stats = calculate_ic_statistics(signal, period_returns, periodicity)
                    
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
        print(f"  (Absolute: score[t] → return[t→t+{period}])")
        print(f"  (Relative: score change[t-{period}→t] → return[t→t+{period}])")
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


#%%
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

#%%

prices=pd.read_excel('ProcessedInputs/Price.xlsx', index_col=0,parse_dates=True)
normalized_score=pd.read_excel('Normalized_Scores/NormalizedScores_World_Scenario_3_Relative_Focus_Scenario_A_Balanced.xlsx', index_col=0,parse_dates=True)
#%%
period_returns_df=calculate_period_returns(prices, 63)

signal_df=calculate_signal(normalized_score,63,'relative')
#%%
results=calculate_ic_statistics(signal_df, period_returns_df, periodicity=63)
#%%
signal_df.shape