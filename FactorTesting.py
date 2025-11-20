"""
FactorTesting Module

This module tests different factor combinations and weighting schemes for country rotation strategies.
It uses FactorTransformer to compute normalized scores with various configurations.
"""

import os
import pandas as pd
import numpy as np
from FactorTransformer import FactorTransformer
from ProcessData import ProcessData

# ============================================================================
# FACTOR SELECTION - Modify these factors as needed
# ============================================================================
selected_factors = [
    'Debt', 'AssetsEquity', 'DvdYieldTTMSpread',
    'EbitdaMargin', 'ROE', 'Equity', 'EV', 'EV_EBIT',
    'Net_Debt_Ebitda', 'Flows', 'Fwd_EV_EBITDA',
    'CashFlowYieldFWDSpread', 'Fwd_PS', 'Fwd_ROE',
    'GDP', 'M2', 'CF', 'Revenue', 'SI_Ratio', 'Ten_Year',
    'ConsensusSalesGrowth', 'ConsensusEbitdaGrowth',
    'ConsensusEarningsGrowth', 'ConsensusCashFlowGrowth',
    'EarningsYieldTTMSpread', 'NetMargin', 'FwdRevenue',
    'FwdEBITDAMargin', 'RollingEarnings', 'FwdRollingEarnings',
    'RollingVol', 'CumFlow'
]

# ============================================================================
# MARKET CONFIGURATION
# ============================================================================
markets = ['World', 'DM', 'EM', 'Asia', 'Europe', 'LatAm']  # List of markets to test

# ============================================================================
# LOAD AND PROCESS DATA (Once for all markets)
# ============================================================================
print("=" * 80)
print("FACTOR TESTING PIPELINE - MULTI-MARKET")
print("=" * 80)

print("\nStep 1: Loading and processing data...")
processor = ProcessData(
    inputs_folder='Inputs',
    classification_file='Classification.xlsx',
    target_date='2010-01-01',
    columns_to_drop=['Saudi Arabia']
)

# Run full data processing pipeline (only once)
processed_data, regions_dict, classification = processor.run_full_pipeline(export_data=False)

print(f"\nData processing completed:")
print(f"  • Processed datasets: {len(processed_data)}")
print(f"  • Markets to test: {len(markets)} - {markets}")
print(f"  • Selected factors: {len(selected_factors)}")

# ============================================================================
# FILTER FACTORS (Once for all markets)
# ============================================================================
print("\nStep 2: Filtering factors...")

# Filter processed data to only include selected factors
filtered_factor_dfs = {
    factor: df for factor, df in processed_data.items() 
    if factor in selected_factors
}

print(f"  • Available factors: {len(filtered_factor_dfs)}/{len(selected_factors)}")

# Check for missing factors
missing_factors = set(selected_factors) - set(filtered_factor_dfs.keys())
if missing_factors:
    print(f"  ⚠ Warning: Missing factors: {missing_factors}")

# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================
print("\nStep 3: Defining test scenarios...")

# Define different weighting scenarios for metrics
metric_weight_scenarios = {
    'Scenario_1_Equal': {
        'zscore': 0.25,
        'absolute_pct': 0.25,
        'relative_rank': 0.25,
        'delta_pct': 0.25
    },
    'Scenario_2_ZScore_Heavy': {
        'zscore': 0.50,
        'absolute_pct': 0.20,
        'relative_rank': 0.20,
        'delta_pct': 0.10
    },
    'Scenario_3_Relative_Focus': {
        'zscore': 0.15,
        'absolute_pct': 0.15,
        'relative_rank': 0.50,
        'delta_pct': 0.20
    },
    'Scenario_4_Momentum_Heavy': {
        'zscore': 0.20,
        'absolute_pct': 0.20,
        'relative_rank': 0.20,
        'delta_pct': 0.40
    },
    'Scenario_5_Historical_Focus': {
        'zscore': 0.30,
        'absolute_pct': 0.40,
        'relative_rank': 0.20,
        'delta_pct': 0.10
    }
}

# Define different category weighting scenarios
category_weight_scenarios = {
    'Scenario_A_Balanced': {
        'Quality': 0.25,
        'Valuation': 0.25,
        'Profitability': 0.25,
        'Momentum': 0.25
    },
    'Scenario_B_Value_Quality': {
        'Quality': 0.35,
        'Valuation': 0.35,
        'Profitability': 0.15,
        'Momentum': 0.15
    },
    'Scenario_C_Growth_Momentum': {
        'Quality': 0.15,
        'Valuation': 0.15,
        'Profitability': 0.30,
        'Momentum': 0.40
    },
    'Scenario_D_Quality_Heavy': {
        'Quality': 0.50,
        'Valuation': 0.20,
        'Profitability': 0.20,
        'Momentum': 0.10
    },
    'Scenario_E_Valuation_Heavy': {
        'Quality': 0.20,
        'Valuation': 0.50,
        'Profitability': 0.15,
        'Momentum': 0.15
    }
}

print(f"  • Metric weight scenarios: {len(metric_weight_scenarios)}")
print(f"  • Category weight scenarios: {len(category_weight_scenarios)}")
print(f"  • Total combinations per market: {len(metric_weight_scenarios) * len(category_weight_scenarios)}")

# ============================================================================
# INITIALIZE STORAGE FOR ALL MARKETS
# ============================================================================
# Dictionary to store all results across all markets
all_normalized_scores = {}  # {market: {scenario_key: df}}
all_composite_scores = {}   # {market: {scenario_key: df}}
all_category_contributions = {}  # {market: {scenario_key: {category: df}}}

# ============================================================================
# LOOP OVER ALL MARKETS
# ============================================================================
print("\n" + "=" * 80)
print("Step 4: Processing all markets...")
print("=" * 80)

for market_idx, market in enumerate(markets, 1):
    print(f"\n{'='*80}")
    print(f"MARKET {market_idx}/{len(markets)}: {market}")
    print(f"{'='*80}")
    
    # Initialize storage for this market
    all_normalized_scores[market] = {}
    all_composite_scores[market] = {}
    all_category_contributions[market] = {}
    
    # ============================================================================
    # RUN FACTOR TRANSFORMATION PIPELINE FOR CURRENT MARKET
    # ============================================================================
    print(f"\nInitializing FactorTransformer for {market} market...")
    
    # Initialize FactorTransformer with selected market
    factor_transformer = FactorTransformer(country_filter=market)
    
    # Transform all factors into metrics
    print(f"Transforming factors into metrics (zscore, absolute_pct, relative_rank, delta_pct)...")
    factor_results = factor_transformer.transform_all(filtered_factor_dfs)
    print(f"  ✓ Transformed {len(factor_results)} factors for {market}")
    
    # ============================================================================
    # GENERATE NORMALIZED SCORES FOR ALL SCENARIO COMBINATIONS
    # ============================================================================
    print(f"\nGenerating normalized scores for all scenarios in {market} market...")

    scenario_count = 0
    total_scenarios = len(metric_weight_scenarios) * len(category_weight_scenarios)
    
    for metric_scenario_name, metric_weights in metric_weight_scenarios.items():
        for category_scenario_name, category_weights in category_weight_scenarios.items():
            scenario_count += 1
            scenario_key = f"{metric_scenario_name}_{category_scenario_name}"
            
            if scenario_count <= 3 or scenario_count == total_scenarios:
                print(f"  Processing scenario {scenario_count}/{total_scenarios}: {scenario_key}")
            elif scenario_count == 4:
                print(f"  ... processing remaining scenarios ...")
            
            # Step 5.1: Calculate weighted average scores
            weighted_scores = factor_transformer.calculate_weighted_average(
                factor_results, 
                metric_weights
            )
            
            # Step 5.2: Aggregate by category
            category_scores, country_scores = factor_transformer.aggregate_by_category(weighted_scores)
            
            # Step 5.3: Calculate composite scores
            composite_scores, contributions = factor_transformer.calculate_composite_score(category_weights)
            
            # Step 5.4: Normalize and rebase contributions
            normalized_scores, rebased_contrib_country, rebased_contrib_factor = \
                factor_transformer.normalize_and_rebase_contributions()
            
            # Store results for this market
            all_normalized_scores[market][scenario_key] = normalized_scores
            all_composite_scores[market][scenario_key] = composite_scores
            all_category_contributions[market][scenario_key] = rebased_contrib_factor
    
    print(f"\n  ✓ Completed {total_scenarios} scenarios for {market}")
    print(f"  ✓ Countries in {market}: {len(normalized_scores.columns)}")
    print(f"  ✓ Date range: {normalized_scores.index[0]} to {normalized_scores.index[-1]}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FACTOR TESTING COMPLETED - ALL MARKETS")
print("=" * 80)

# Calculate total scenarios across all markets
total_scenarios_all_markets = sum(len(scenarios) for scenarios in all_normalized_scores.values())

print(f"\nSummary:")
print(f"  • Markets tested: {len(markets)}")
print(f"  • Scenarios per market: {len(metric_weight_scenarios) * len(category_weight_scenarios)}")
print(f"  • Total scenario-market combinations: {total_scenarios_all_markets}")
print(f"  • Factors used: {len(filtered_factor_dfs)}")

print(f"\nBreakdown by market:")
for market in markets:
    sample_df = list(all_normalized_scores[market].values())[0]
    print(f"  • {market}: {len(all_normalized_scores[market])} scenarios, "
          f"{len(sample_df.columns)} countries, "
          f"{len(sample_df)} dates")

# ============================================================================
# EXPORT NORMALIZED SCORES TO EXCEL
# ============================================================================
print("\n" + "=" * 80)
print("Step 5: Exporting normalized scores to Excel...")
print("=" * 80)

output_folder = 'Normalized_Scores'
os.makedirs(output_folder, exist_ok=True)

export_count = 0
total_exports = sum(len(scenarios) for scenarios in all_normalized_scores.values())

for market in markets:
    print(f"\nExporting {market} market...")
    market_export_count = 0
    
    for scenario_key, normalized_scores_df in all_normalized_scores[market].items():
        try:
            # Create filename with market suffix
            filename = f"NormalizedScores_{market}_{scenario_key}.xlsx"
            output_path = os.path.join(output_folder, filename)
            
            # Export to Excel
            normalized_scores_df.to_excel(output_path, index=True)
            export_count += 1
            market_export_count += 1
            
            if market_export_count <= 2:  # Only show first 2 exports per market
                print(f"  ✓ Exported: {filename}")
            elif market_export_count == 3:
                print(f"  ... exporting remaining {len(all_normalized_scores[market]) - 2} scenarios for {market} ...")
            
        except Exception as e:
            print(f"  ✗ Failed to export {market}_{scenario_key}: {str(e)}")
    
    print(f"  ✓ Completed {market_export_count} exports for {market}")

print(f"\n{'='*80}")
print(f"✓ Successfully exported {export_count}/{total_exports} total files")
print(f"✓ Export folder: {output_folder}/")
print(f"{'='*80}")

print("\n" + "=" * 80)
print("DATA STRUCTURES AVAILABLE:")
print("=" * 80)
print("\n1. all_normalized_scores: Dict[market, Dict[scenario_key, normalized_scores_df]]")
print("   - Cross-sectionally normalized composite scores (0 to 1)")
print("   - Structure: {market: {scenario: df}}")
print("   - DataFrame: Index=dates, Columns=countries")

print("\n2. all_composite_scores: Dict[market, Dict[scenario_key, composite_scores_df]]")
print("   - Raw composite scores before normalization")
print("   - Structure: {market: {scenario: df}}")
print("   - DataFrame: Index=dates, Columns=countries")

print("\n3. all_category_contributions: Dict[market, Dict[scenario_key, Dict[category, contributions_df]]]")
print("   - Rebased contributions by category")
print("   - Structure: {market: {scenario: {category: df}}}")
print("   - DataFrame: Index=dates, Columns=countries")

print("\n4. markets: List of markets tested")
print(f"   - {markets}")

print("\n5. Total files exported:")
print(f"   - {export_count} Excel files in {output_folder}/")

print("\n" + "=" * 80)
print("EXAMPLE USAGE:")
print("=" * 80)
print("""
# Access a specific market and scenario's results:
market = 'World'
scenario_name = 'Scenario_1_Equal_Scenario_A_Balanced'
scores = all_normalized_scores[market][scenario_name]

# Get latest rankings for a market:
latest_date = scores.index[-1]
latest_rankings = scores.loc[latest_date].sort_values(ascending=False)
print(f"Latest rankings for {market}:")
print(latest_rankings)

# Compare scenarios within a market:
scenario_1 = all_normalized_scores['World']['Scenario_1_Equal_Scenario_A_Balanced']
scenario_2 = all_normalized_scores['World']['Scenario_2_ZScore_Heavy_Scenario_B_Value_Quality']
correlation = scenario_1.corrwith(scenario_2, axis=1).mean()
print(f"Average correlation: {correlation:.4f}")

# Compare same scenario across markets:
world_scores = all_normalized_scores['World']['Scenario_1_Equal_Scenario_A_Balanced']
dm_scores = all_normalized_scores['DM']['Scenario_1_Equal_Scenario_A_Balanced']

# Access category contributions:
contributions = all_category_contributions['World']['Scenario_1_Equal_Scenario_A_Balanced']
quality_contrib = contributions['Quality']

# Iterate over all markets:
for market in markets:
    print(f"\\nMarket: {market}")
    for scenario_key in list(all_normalized_scores[market].keys())[:3]:
        print(f"  Scenario: {scenario_key}")
""")

print("\n" + "=" * 80)

# ============================================================================
# OPTIONAL: QUICK ANALYSIS
# ============================================================================
print("\nPerforming quick analysis across markets...")

# Get latest date from first market
sample_market = markets[0]
latest_date = list(all_normalized_scores[sample_market].values())[0].index[-1]

print(f"\nLatest date: {latest_date}")
print("\nTop 5 countries by market (using Scenario_1_Equal_Scenario_A_Balanced):")

for market in markets:
    scenario_key = 'Scenario_1_Equal_Scenario_A_Balanced'
    if scenario_key in all_normalized_scores[market]:
        scores = all_normalized_scores[market][scenario_key].loc[latest_date].sort_values(ascending=False)
        print(f"\n{market} Market:")
        print(scores.head())

print("\n" + "=" * 80)
print(f"Ready for further analysis!")
print(f"Total combinations available: {len(markets)} markets × {len(metric_weight_scenarios) * len(category_weight_scenarios)} scenarios = {total_scenarios_all_markets} datasets")
print("=" * 80)

