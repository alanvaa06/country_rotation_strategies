"""
Country Rotation Strategy - Main Script

This script implements a quantitative country rotation strategy using financial and economic data.
It processes multiple datasets, performs regional aggregations, calculates derived metrics,
and exports the results for further analysis.

All functions are imported from function_module.py for better code organization.
The FactorTransformer class is imported from FactorTransformer.py.
"""

import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from FactorTransformer import FactorTransformer
from ProcessData import ProcessData

warnings.filterwarnings('ignore')

# Set working directory
#os.chdir('D:/Users/avazquez/OneDrive - valmexcasabolsa/Documents/QuantModels/country_rotation')

# Configure plotting style
plt.style.use('seaborn-v0_8')

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
        # Initialize ProcessData instance
        processor = ProcessData(
            inputs_folder='Inputs',
            classification_file='Classification.xlsx',
            target_date='2010-01-01',
            columns_to_drop=['Saudi Arabia']
        )
        
        # Run full pipeline with export enabled
        processed_data, regions_dict, classification = processor.run_full_pipeline(export_data=False)
        
        # Show some key metrics that were created
        # Get original file names (before derived metrics were added)
        file_names = processor.original_file_names
        derived_metrics = [k for k in processed_data.keys() if k not in file_names]
        if derived_metrics:
            print(f"\n🔧 NEW DERIVED METRICS ({len(derived_metrics)}):")
            for i, metric in enumerate(derived_metrics[:10], 1):  # Show first 10
                print(f"   {i}. {metric}")
            if len(derived_metrics) > 10:
                print(f"   ... and {len(derived_metrics) - 10} more derived metrics")
        
        print(f"\n✅ All processed data exported to: ProcessedInputs/")
        print(f"✅ Ready for quantitative analysis and strategy implementation!")
        
        return processed_data, regions_dict, classification, processor
        
    except Exception as e:
        print(f"\n❌ ERROR IN PIPELINE EXECUTION:")
        print(f"   {str(e)}")
        print(f"\n🔧 Please check your data files and try again.")
        raise


# Execute the main pipeline
processed_data, regions_dict, classification, processor = run_inputs()

# Make key variables available in the global scope for interactive use
dataFrames = processed_data

print(f"\n📋 VARIABLES AVAILABLE FOR ANALYSIS:")
print(f"   • dataFrames: {len(dataFrames)} processed datasets")
print(f"   • regions_dict: Regional country groupings")
print(f"   • classification: Country classification DataFrame")
print(f"   • processed_data: Same as dataFrames (alias)")

# Quick data exploration
processor.explore_data()

#%%

# ============================================================================
# FACTOR TRANSFORMATION AND ANALYSIS
# ============================================================================

# Prepare factor dataframes for transformation
factor_dfs = dataFrames.copy()

# ============================================================================
# FACTOR ANALYSIS CONFIGURATION
# ============================================================================

# Define weights for metric combinations
weights = {
    'zscore': 0.25,
    'absolute_pct': 0.25,
    'relative_rank': 0.1,
    'delta_pct': 0.4
}

# Define weights for category combinations
category_weights = {
    'Quality': 0.25,
    'Valuation': 0.1,
    'Profitability': 0.35,
    'Momentum': 0.3
}
#%%
# ============================================================================
# PHASE 1: INITIAL ANALYSIS (Full Factor Set)
# ============================================================================

# Step 1: Initialize FactorTransformer with all factors
factor_transformer = FactorTransformer('World')

# Step 2: Transform all factors
factor_results = factor_transformer.transform_all(factor_dfs)

# Step 3: Calculate weighted averages
weighted_scores = factor_transformer.calculate_weighted_average(factor_results, weights)

category_scores, country_scores = factor_transformer.aggregate_by_category(weighted_scores)

composite_scores, contributions = factor_transformer.calculate_composite_score(category_weights)

normalized_scores, rebased_by_country, rebased_by_factor = factor_transformer.normalize_and_rebase_contributions()

# Step 4: Analyze factor redundancy
factor_redundancy_results = factor_transformer.analyze_factor_redundancy(
    distance_threshold=0.5,
    linkage_method='ward',
    selection_criterion='unique' # or 'unique' or 'central'
)

# Step 5: Visualize clusters
factor_transformer.plot_factor_dendrogram()

# Step 6: Review recommendations
print("\nRecommended factors:")
for factor in factor_redundancy_results['recommended_factors']:
    category = factor_transformer.factor_map[factor]
    print(f"  {factor} ({category}): {factor_redundancy_results['selection_rationale'][factor]}")

#%%
# ============================================================================
# PHASE 2: FILTERED PIPELINE (Selected Factors)
# ============================================================================

# Step 7: Filter raw data to selected factors
selected_factors = factor_redundancy_results['recommended_factors']
filtered_factor_dfs = {
    factor: df for factor, df in factor_dfs.items() 
    if factor in selected_factors
}

# Step 8: Re-initialize FactorTransformer for reduced factor set
factor_transformer_reduced = FactorTransformer('World')

# Step 9: Re-run entire pipeline with filtered factors
factor_results_reduced = factor_transformer_reduced.transform_all(filtered_factor_dfs)

weighted_scores_reduced = factor_transformer_reduced.calculate_weighted_average(
    factor_results_reduced, 
    weights
)

category_scores_reduced, country_scores_reduced = factor_transformer_reduced.aggregate_by_category(
    weighted_scores_reduced
)

composite_scores_reduced, contributions_reduced = factor_transformer_reduced.calculate_composite_score(
    category_weights
)

normalized_scores_reduced, rebased_by_country, rebased_by_factor = factor_transformer_reduced.normalize_and_rebase_contributions()

factor_changes_reduced, total_changes_reduced = factor_transformer_reduced.calculate_factor_contribution_changes(period=5)

# Step 10: Plot results
factor_transformer_reduced.plot_factor_contributions()

#%%
# Plot normalized scores for each country
for country in normalized_scores.columns:
    normalized_scores[country].plot(title=country.upper(), figsize=(12, 6))
    plt.show()

#%%
for date in factor_transformer_reduced.change_dates[-3:]:
    factor_transformer_reduced.plot_factor_contributions(date)
#%%
)