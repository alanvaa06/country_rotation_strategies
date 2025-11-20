"""
Country Rotation Strategy - Simplified Backtest Script

This script implements a streamlined backtesting workflow:
    1. Data Processing - Load and process raw financial/economic data using ProcessData
    2. Load Normalized Scores - Import pre-calculated normalized scores from file
    3. Backtesting - Run strategy backtests with various configurations
    4. Performance Analysis - Compare results across strategies

Author: Investment Strategy Team
Last Updated: November 2025
"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================

# Standard library imports
import os
import warnings

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from ProcessData import ProcessData
from backtest import Backtest

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


# ============================================================================
# STEP 1: DATA PROCESSING
# ============================================================================

print("=" * 80)
print("COUNTRY ROTATION STRATEGY - BACKTEST PIPELINE")
print("=" * 80)

print("\n" + "=" * 80)
print("STEP 1: DATA PROCESSING")
print("=" * 80)

# Initialize ProcessData instance
processor = ProcessData(
    inputs_folder='Inputs',
    classification_file='Classification.xlsx',
    target_date='2010-01-01',
    columns_to_drop=['Saudi Arabia']
)

# Run full pipeline (no export needed since we're loading normalized scores)
processed_data, regions_dict, classification = processor.run_full_pipeline(export_data=False)

print(f"\nData processing completed successfully!")
print(f"   • Processed datasets: {len(processed_data)}")
print(f"   • Countries in classification: {len(classification)}")

# Extract price data for backtesting
prices = processed_data['Price']
print(f"   • Price data shape: {prices.shape}")
print(f"   • Date range: {prices.index[0]} to {prices.index[-1]}")

#%%
# ============================================================================
# STEP 2: LOAD NORMALIZED SCORES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: LOADING NORMALIZED SCORES")
print("=" * 80)

# Define normalized scores file path
normalized_scores_folder = 'Normalized_Scores'
normalized_scores_file = 'NormalizedScores_World_Scenario_5_Historical_Focus_Scenario_D_Quality_Heavy.xlsx'
normalized_scores_path = os.path.join(normalized_scores_folder, normalized_scores_file)

# Load normalized scores
try:
    print(f"\nLoading normalized scores from: {normalized_scores_path}")
    normalized_scores = pd.read_excel(
        normalized_scores_path,
        index_col=0,
        parse_dates=True
    )
    
    print(f"✓ Normalized scores loaded successfully!")
    print(f"   • Shape: {normalized_scores.shape}")
    print(f"   • Date range: {normalized_scores.index[0]} to {normalized_scores.index[-1]}")
    print(f"   • Countries: {len(normalized_scores.columns)}")
    print(f"   • Sample countries: {list(normalized_scores.columns[:5])}")
    
except FileNotFoundError:
    print(f"\n⚠ ERROR: File not found: {normalized_scores_path}")
    print(f"   Please ensure the file exists in the {normalized_scores_folder} folder.")
    raise
except Exception as e:
    print(f"\n⚠ ERROR loading normalized scores: {str(e)}")
    raise


# ============================================================================
# STEP 3: PREPARE DATA FOR BACKTESTING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: PREPARING DATA FOR BACKTESTING")
print("=" * 80)

# Verify data alignment
common_dates = normalized_scores.index.intersection(prices.index)
print(f"\nData alignment check:")
print(f"   • Common dates: {len(common_dates)}")
print(f"   • Date overlap: {common_dates[0]} to {common_dates[-1]}")

# Check if Benchmark column exists in prices
if 'Benchmark' in prices.columns:
    print(f"   ✓ Benchmark column found in prices")
    benchmark_col = 'Benchmark'
elif 'World' in prices.columns:
    print(f"   ⚠ Using 'World' as benchmark (no 'Benchmark' column found)")
    benchmark_col = 'World'
    prices['Benchmark'] = prices['World']
else:
    raise ValueError("No benchmark column found in prices. Expected 'Benchmark' or 'World'.")


# ============================================================================
# STEP 4: RUN BACKTESTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: RUNNING BACKTESTS WITH DIFFERENT CONFIGURATIONS")
print("=" * 80)

# Dictionary to store all backtest instances
backtests = {}

# Configuration parameters
PERIODICITY = 5  # Rebalancing frequency in days
TRANSACTION_COST_BPS = 2.0  # Transaction cost in basis points


# ----------------------------------------------------------------------------
# Test 1: Absolute Selection (0.75) + Equal Weighting + 100% Active
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("TEST 1: Absolute (0.75) | Equal Weight | 100% Active")
print("-" * 80)

bt_abs_equal_100 = Backtest(
    normalized_score=normalized_scores,
    prices=prices,
    selection_criteria="absolute",
    absolute_selection_score=0.75,
    weighting_method="Equal",
    bmk=benchmark_col,
    bmk_weight=0.0,
    periodicity=PERIODICITY,
    transaction_cost_bps=TRANSACTION_COST_BPS
)

results_1 = bt_abs_equal_100.run_backtest()
summary_1 = bt_abs_equal_100.get_performance_summary()
print("\nPerformance Summary:")
print(summary_1.round(4))

backtests['Absolute_0.75_Equal_100'] = bt_abs_equal_100


# ----------------------------------------------------------------------------
# Test 2: Absolute Selection (0.75) + Risk Parity + 100% Active
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("TEST 2: Absolute (0.75) | Risk Parity | 100% Active")
print("-" * 80)

bt_abs_rp_100 = Backtest(
    normalized_score=normalized_scores,
    prices=prices,
    selection_criteria="absolute",
    absolute_selection_score=0.75,
    weighting_method="Risk_Parity",
    bmk=benchmark_col,
    bmk_weight=0.0,
    periodicity=PERIODICITY,
    transaction_cost_bps=TRANSACTION_COST_BPS,
    risk_parity_lookback=60
)

results_2 = bt_abs_rp_100.run_backtest()
summary_2 = bt_abs_rp_100.get_performance_summary()
print("\nPerformance Summary:")
print(summary_2.round(4))

backtests['Absolute_0.75_RiskParity_100'] = bt_abs_rp_100


# ----------------------------------------------------------------------------
# Test 3: Absolute Selection (0.60) + Equal Weighting + 100% Active
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("TEST 3: Absolute (0.60) | Equal Weight | 100% Active")
print("-" * 80)

bt_abs_equal_100_low = Backtest(
    normalized_score=normalized_scores,
    prices=prices,
    selection_criteria="absolute",
    absolute_selection_score=0.60,
    weighting_method="Equal",
    bmk=benchmark_col,
    bmk_weight=0.0,
    periodicity=PERIODICITY,
    transaction_cost_bps=TRANSACTION_COST_BPS
)

results_3 = bt_abs_equal_100_low.run_backtest()
summary_3 = bt_abs_equal_100_low.get_performance_summary()
print("\nPerformance Summary:")
print(summary_3.round(4))

backtests['Absolute_0.60_Equal_100'] = bt_abs_equal_100_low


# ----------------------------------------------------------------------------
# Test 4: Absolute Selection (0.75) + Equal Weighting + 70/30 Blend
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("TEST 4: Absolute (0.75) | Equal Weight | 70/30 Blend")
print("-" * 80)

bt_abs_equal_70 = Backtest(
    normalized_score=normalized_scores,
    prices=prices,
    selection_criteria="absolute",
    absolute_selection_score=0.75,
    weighting_method="Equal",
    bmk=benchmark_col,
    bmk_weight=0.3,
    periodicity=PERIODICITY,
    transaction_cost_bps=TRANSACTION_COST_BPS
)

results_4 = bt_abs_equal_70.run_backtest()
summary_4 = bt_abs_equal_70.get_performance_summary()
print("\nPerformance Summary:")
print(summary_4.round(4))

backtests['Absolute_0.75_Equal_70-30'] = bt_abs_equal_70


# ----------------------------------------------------------------------------
# Test 5: Relative Selection (Top 5) + Equal Weighting + 100% Active
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("TEST 5: Relative (Top 5) | Equal Weight | 100% Active")
print("-" * 80)

bt_rel_equal_100 = Backtest(
    normalized_score=normalized_scores,
    prices=prices,
    selection_criteria="relative",
    relative_selection_score=5,
    weighting_method="Equal",
    bmk=benchmark_col,
    bmk_weight=0.0,
    periodicity=PERIODICITY,
    transaction_cost_bps=TRANSACTION_COST_BPS
)

results_5 = bt_rel_equal_100.run_backtest()
summary_5 = bt_rel_equal_100.get_performance_summary()
print("\nPerformance Summary:")
print(summary_5.round(4))

backtests['Relative_Top5_Equal_100'] = bt_rel_equal_100


# ----------------------------------------------------------------------------
# Test 6: Relative Selection (Top 5) + Risk Parity + 100% Active
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("TEST 6: Relative (Top 5) | Risk Parity | 100% Active")
print("-" * 80)

bt_rel_rp_100 = Backtest(
    normalized_score=normalized_scores,
    prices=prices,
    selection_criteria="relative",
    relative_selection_score=5,
    weighting_method="Risk_Parity",
    bmk=benchmark_col,
    bmk_weight=0.0,
    periodicity=PERIODICITY,
    transaction_cost_bps=TRANSACTION_COST_BPS,
    risk_parity_lookback=60
)

results_6 = bt_rel_rp_100.run_backtest()
summary_6 = bt_rel_rp_100.get_performance_summary()
print("\nPerformance Summary:")
print(summary_6.round(4))

backtests['Relative_Top5_RiskParity_100'] = bt_rel_rp_100


# ----------------------------------------------------------------------------
# Test 7: Relative Selection (Top 3) + Equal Weighting + 100% Active
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("TEST 7: Relative (Top 3) | Equal Weight | 100% Active")
print("-" * 80)

bt_rel_equal_100_top3 = Backtest(
    normalized_score=normalized_scores,
    prices=prices,
    selection_criteria="relative",
    relative_selection_score=3,
    weighting_method="Equal",
    bmk=benchmark_col,
    bmk_weight=0.0,
    periodicity=PERIODICITY,
    transaction_cost_bps=TRANSACTION_COST_BPS
)

results_7 = bt_rel_equal_100_top3.run_backtest()
summary_7 = bt_rel_equal_100_top3.get_performance_summary()
print("\nPerformance Summary:")
print(summary_7.round(4))

backtests['Relative_Top3_Equal_100'] = bt_rel_equal_100_top3


# ----------------------------------------------------------------------------
# Test 8: Relative Selection (Top 5) + Equal Weighting + 50/50 Blend
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("TEST 8: Relative (Top 5) | Equal Weight | 50/50 Blend")
print("-" * 80)

bt_rel_equal_50 = Backtest(
    normalized_score=normalized_scores,
    prices=prices,
    selection_criteria="relative",
    relative_selection_score=5,
    weighting_method="Equal",
    bmk=benchmark_col,
    bmk_weight=0.5,
    periodicity=PERIODICITY,
    transaction_cost_bps=TRANSACTION_COST_BPS
)

results_8 = bt_rel_equal_50.run_backtest()
summary_8 = bt_rel_equal_50.get_performance_summary()
print("\nPerformance Summary:")
print(summary_8.round(4))

backtests['Relative_Top5_Equal_50-50'] = bt_rel_equal_50


# ============================================================================
# STEP 5: COMPARATIVE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: COMPARATIVE ANALYSIS - ALL STRATEGIES")
print("=" * 80)

# Create comparison DataFrame
comparison_data = []
for name, bt in backtests.items():
    summary = bt.get_performance_summary()
    comparison_data.append({
        'Strategy': name,
        'Total Return': summary.loc['Portfolio', 'Total Return'],
        'Ann. Return': summary.loc['Portfolio', 'Annualized Return'],
        'Ann. Volatility': summary.loc['Portfolio', 'Annualized Volatility'],
        'Sharpe Ratio': summary.loc['Portfolio', 'Sharpe Ratio'],
        'Max Drawdown': summary.loc['Portfolio', 'Max Drawdown'],
        'Win Rate': summary.loc['Portfolio', 'Win Rate'],
        'Active Return': summary.loc['Active', 'Total Return'],
        'Info Ratio': summary.loc['Active', 'Information Ratio'],
        'Avg Turnover': bt.turnover.mean(),
        'Avg TC (bps)': bt.transaction_costs.mean() * 10000
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Sharpe Ratio', ascending=False)

print("\nSTRATEGY COMPARISON (Sorted by Sharpe Ratio):")
print("=" * 80)
print(comparison_df.to_string(index=False))

# Identify best strategies
best_sharpe = comparison_df.iloc[0]
best_return = comparison_df.loc[comparison_df['Total Return'].idxmax()]
best_info_ratio = comparison_df.loc[comparison_df['Info Ratio'].idxmax()]

print("\n" + "=" * 80)
print("BEST PERFORMING STRATEGIES:")
print("=" * 80)
print(f"\nHighest Sharpe Ratio:")
print(f"   Strategy: {best_sharpe['Strategy']}")
print(f"   Sharpe Ratio: {best_sharpe['Sharpe Ratio']:.4f}")
print(f"   Annualized Return: {best_sharpe['Ann. Return']:.2%}")
print(f"   Annualized Volatility: {best_sharpe['Ann. Volatility']:.2%}")

print(f"\nHighest Total Return:")
print(f"   Strategy: {best_return['Strategy']}")
print(f"   Total Return: {best_return['Total Return']:.2%}")
print(f"   Sharpe Ratio: {best_return['Sharpe Ratio']:.4f}")

print(f"\nHighest Information Ratio:")
print(f"   Strategy: {best_info_ratio['Strategy']}")
print(f"   Information Ratio: {best_info_ratio['Info Ratio']:.4f}")
print(f"   Active Return: {best_info_ratio['Active Return']:.2%}")


# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: GENERATING VISUALIZATIONS")
print("=" * 80)

# Plot 1: Comparative Cumulative Returns
print("\nGenerating comparative cumulative returns plot...")
fig, ax = plt.subplots(figsize=(16, 8))

for name, bt in backtests.items():
    cum_returns = (1 + bt.returns).cumprod()
    ax.plot(cum_returns.index, cum_returns.values, label=name, linewidth=2, alpha=0.8)

# Add benchmark
benchmark_returns = backtests[list(backtests.keys())[0]].benchmark_returns
benchmark_cum = (1 + benchmark_returns).cumprod()
ax.plot(benchmark_cum.index, benchmark_cum.values, 
        label='Benchmark', linewidth=3, linestyle='--', color='black', alpha=0.7)

ax.set_title('Cumulative Returns: All Strategies vs Benchmark', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return', fontsize=12)
ax.legend(loc='best', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Plot 2: Performance Metrics Comparison
print("\nGenerating performance metrics comparison...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

strategies_short = [name.replace('_', '\n') for name in comparison_df['Strategy']]

# Sharpe Ratio
colors_sharpe = ['green' if x > 0 else 'red' for x in comparison_df['Sharpe Ratio']]
ax1.barh(range(len(strategies_short)), comparison_df['Sharpe Ratio'], color=colors_sharpe, alpha=0.7)
ax1.set_yticks(range(len(strategies_short)))
ax1.set_yticklabels(strategies_short, fontsize=8)
ax1.set_xlabel('Sharpe Ratio')
ax1.set_title('Sharpe Ratio by Strategy', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

# Total Return
colors_return = ['green' if x > 0 else 'red' for x in comparison_df['Total Return']]
ax2.barh(range(len(strategies_short)), comparison_df['Total Return'], color=colors_return, alpha=0.7)
ax2.set_yticks(range(len(strategies_short)))
ax2.set_yticklabels(strategies_short, fontsize=8)
ax2.set_xlabel('Total Return')
ax2.set_title('Total Return by Strategy', fontweight='bold')
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

# Information Ratio
colors_ir = ['green' if x > 0 else 'red' for x in comparison_df['Info Ratio']]
ax3.barh(range(len(strategies_short)), comparison_df['Info Ratio'], color=colors_ir, alpha=0.7)
ax3.set_yticks(range(len(strategies_short)))
ax3.set_yticklabels(strategies_short, fontsize=8)
ax3.set_xlabel('Information Ratio')
ax3.set_title('Information Ratio by Strategy', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')
ax3.invert_yaxis()

# Average Turnover
ax4.barh(range(len(strategies_short)), comparison_df['Avg Turnover'], color='steelblue', alpha=0.7)
ax4.set_yticks(range(len(strategies_short)))
ax4.set_yticklabels(strategies_short, fontsize=8)
ax4.set_xlabel('Average Turnover')
ax4.set_title('Average Turnover by Strategy', fontweight='bold')
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
ax4.grid(True, alpha=0.3, axis='x')
ax4.invert_yaxis()

plt.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()


# ============================================================================
# STEP 7: DETAILED ANALYSIS FOR BEST STRATEGY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: DETAILED ANALYSIS FOR BEST STRATEGY")
print("=" * 80)

# Select best strategy by Sharpe Ratio
best_strategy_name = best_sharpe['Strategy']
best_bt = backtests[best_strategy_name]

print(f"\nAnalyzing: {best_strategy_name}")
print("=" * 80)

# Plot cumulative returns
print("\nGenerating cumulative returns plot...")
best_bt.plot_cumulative_returns()

# Plot weights over time
print("\nGenerating portfolio weights plot...")
best_bt.plot_weights_over_time(top_n=10)

# Turnover analysis
print("\nPerforming turnover analysis...")
turnover_results = best_bt.portfolio_turnover_analysis(plot=True)

# Performance attribution
print("\nPerforming performance attribution analysis...")
attribution_results = best_bt.performance_attribution_analysis(plot=True)

# IC analysis
print("\nPerforming Information Coefficient (IC) analysis...")
ic_results = best_bt.IC_analysis(rolling_window=20, plot=True)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("BACKTEST PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)

print(f"\nExecuted {len(backtests)} backtest configurations")
print(f"\nBest Strategy (by Sharpe Ratio): {best_strategy_name}")
print(f"   • Sharpe Ratio: {best_sharpe['Sharpe Ratio']:.4f}")
print(f"   • Annualized Return: {best_sharpe['Ann. Return']:.2%}")
print(f"   • Annualized Volatility: {best_sharpe['Ann. Volatility']:.2%}")
print(f"   • Max Drawdown: {best_sharpe['Max Drawdown']:.2%}")
print(f"   • Information Ratio: {best_sharpe['Info Ratio']:.4f}")
print(f"   • Average Turnover: {best_sharpe['Avg Turnover']:.2%}")

print("\n" + "=" * 80)
print("All results are available in the 'backtests' dictionary")
print("Access individual backtests: backtests['<strategy_name>']")
print("=" * 80)
