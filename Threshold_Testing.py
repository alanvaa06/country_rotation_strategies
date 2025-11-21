"""
Threshold Testing Module

This module provides functionality to test different factor reduction thresholds
and analyze their impact on the final composite score's rank correlation.

The module can be run independently from strategy.py to test various threshold values
and visualize the relationship between threshold, number of factors, and correlation.

Author: Alan Vazquez, CFA, Master Jedi
Last Updated: November 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from typing import List, Dict, Type

from FactorTransformer import FactorTransformer
from ProcessData import ProcessData

# Set working directory
os.chdir('D:/Users/avazquez/OneDrive - valmexcasabolsa/Documents/QuantModels/country_rotation')

# Configure plotting style
plt.style.use('seaborn-v0_8')


def test_factor_reduction_thresholds(
    factor_transformer_class: Type,
    original_factor_dfs: Dict[str, pd.DataFrame],
    metric_weights: Dict[str, float],
    category_weights: Dict[str, float],
    country_filter: str,
    thresholds_to_test: List[float],
    selection_criterion: str = 'coverage',
    linkage_method: str = 'ward'
) -> pd.DataFrame:
    """
    Tests multiple factor redundancy thresholds and measures the impact on the
    final composite score's rank correlation.

    Args:
        factor_transformer_class: The FactorTransformer class (not an instance).
        original_factor_dfs: The full, original dictionary of factor DataFrames.
        metric_weights: Weights for calculate_weighted_average (e.g., zscore, absolute_pct).
        category_weights: Weights for calculate_composite_score (e.g., Valuation, Quality).
        country_filter: The country group to use (e.g., 'World', 'DM', 'EM').
        thresholds_to_test: A list of distance thresholds to test (e.g., [0.1, 0.3, 0.5]).
        selection_criterion: Criterion for analyze_factor_redundancy ('coverage', 'unique', 'central').
        linkage_method: Linkage method for analyze_factor_redundancy ('ward', 'average').

    Returns:
        A DataFrame with results: [threshold, n_factors, spearman_corr]
    """

    print("--- 1. Calculating Baseline Scores (All Factors) ---")
    
    # --- Run baseline pipeline (all factors) ---
    base_transformer = factor_transformer_class(country_filter)
    base_factor_results = base_transformer.transform_all(original_factor_dfs)
    base_weighted_scores = base_transformer.calculate_weighted_average(
        base_factor_results, metric_weights
    )
    base_transformer.aggregate_by_category(base_weighted_scores)
    base_transformer.calculate_composite_score(category_weights)
    base_normalized_scores, _, _ = base_transformer.normalize_and_rebase_contributions()

    # Get total number of factors available *after* filtering and processing
    total_factors = len(base_transformer.weighted_average_scores)
    print(f"Baseline calculated with {total_factors} factors.")

    # Store results here
    results_list = []

    for thresh in thresholds_to_test:
        print(f"\n--- 2. Testing Threshold: {thresh} ---")

        # --- Step A: Run redundancy analysis ---
        # We use the 'base_transformer' which has all weighted scores calculated
        print(f"  Running redundancy analysis...")
        try:
            redundancy_results = base_transformer.analyze_factor_redundancy(
                distance_threshold=thresh,
                selection_criterion=selection_criterion,
                linkage_method=linkage_method
            )
        except Exception as e:
            print(f"  Error during redundancy analysis for threshold {thresh}: {e}")
            results_list.append({
                'threshold': thresh,
                'n_factors': np.nan,
                'spearman_corr': np.nan,
                'error': str(e)
            })
            continue
            
        recommended_factors = redundancy_results['recommended_factors']
        n_reduced = len(recommended_factors)
        print(f"  Threshold {thresh} selected {n_reduced} factors.")

        # --- Step B: Filter original data ---
        reduced_factor_dfs = {
            k: v for k, v in original_factor_dfs.items()
            if k in recommended_factors
        }

        if not reduced_factor_dfs:
            print(f"  Warning: No factors selected for threshold {thresh}. Skipping.")
            results_list.append({
                'threshold': thresh,
                'n_factors': 0,
                'spearman_corr': np.nan,
                'error': 'No factors selected'
            })
            continue

        # --- Step C: Run *new* pipeline on reduced data ---
        print(f"  Re-running pipeline with {n_reduced} factors...")
        try:
            reduced_transformer = factor_transformer_class(country_filter)
            reduced_results = reduced_transformer.transform_all(reduced_factor_dfs)
            reduced_weighted = reduced_transformer.calculate_weighted_average(
                reduced_results, metric_weights
            )
            reduced_transformer.aggregate_by_category(reduced_weighted)
            reduced_transformer.calculate_composite_score(category_weights)
            reduced_normalized_scores, _, _ = reduced_transformer.normalize_and_rebase_contributions()
        
        except Exception as e:
            print(f"  Error running reduced pipeline for threshold {thresh}: {e}")
            results_list.append({
                'threshold': thresh,
                'n_factors': n_reduced,
                'spearman_corr': np.nan,
                'error': str(e)
            })
            continue

        # --- Step D: Align and Calculate Correlation ---
        # Align DataFrames on index and columns to ensure comparison is 1-to-1
        aligned_base, aligned_reduced = base_normalized_scores.align(
            reduced_normalized_scores,
            join='inner',
            axis=None,
            fill_value=np.nan
        )
        
        base_flat = aligned_base.values.flatten()
        reduced_flat = aligned_reduced.values.flatten()
        
        # Remove NaNs from *both* lists simultaneously
        valid_mask = ~np.isnan(base_flat) & ~np.isnan(reduced_flat)
        base_flat_clean = base_flat[valid_mask]
        reduced_flat_clean = reduced_flat[valid_mask]

        if len(base_flat_clean) < 10:  # Need some data to correlate
             print(f"  Warning: Not enough overlapping data ({len(base_flat_clean)}) to calculate correlation.")
             corr = np.nan
             error_msg = 'Not enough data'
        else:
             corr, _ = spearmanr(base_flat_clean, reduced_flat_clean)
             error_msg = None
             print(f"  Spearman Correlation: {corr:.6f}")

        results_list.append({
            'threshold': thresh,
            'n_factors': n_reduced,
            'spearman_corr': corr,
            'error': error_msg
        })

    print("\n--- 3. Analysis Complete ---")
    return pd.DataFrame(results_list).set_index('threshold')


def plot_threshold_analysis(analysis_results: pd.DataFrame, figsize: tuple = (10, 6)):
    """
    Plots the threshold analysis results showing the relationship between
    threshold, number of factors, and correlation.

    Args:
        analysis_results: DataFrame with results from test_factor_reduction_thresholds
        figsize: Figure size tuple (default: (10, 6))
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot n_factors on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Distance Threshold (1 - |correlation|)')
    ax1.set_ylabel('Number of Recommended Factors', color=color)
    ax1.plot(analysis_results.index, analysis_results['n_factors'], 
             color=color, marker='o', label='Num Factors', linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Create a second y-axis for correlation
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Spearman Rank Correlation (vs. Baseline)', color=color)
    ax2.plot(analysis_results.index, analysis_results['spearman_corr'], 
             color=color, marker='s', label='Rank Corr', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05)  # Correlation is 0-1

    fig.suptitle('Factor Reduction Analysis: Threshold vs. Correlation', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
    
    plt.show()


def load_processed_data():
    """
    Load and process data from scratch for threshold testing.
    This function runs the complete data processing pipeline from the 'Inputs' folder.
    
    Returns:
        tuple: (factor_dfs, weights, category_weights)
    """
    print("=" * 70)
    print("LOADING AND PROCESSING DATA FROM SCRATCH")
    print("=" * 70)
    
    # Initialize ProcessData instance
    processor = ProcessData(
        inputs_folder='Inputs',
        classification_file='Classification.xlsx',
        target_date='2010-01-01',
        columns_to_drop=['Saudi Arabia']
    )
    
    # Run full pipeline with export disabled
    factor_dfs, regions_dict, classification = processor.run_full_pipeline(export_data=False)
    
    print(f"\nProcessing complete: {len(factor_dfs)} factor datasets ready for testing.")
    
    # Define weights
    weights = {
        'zscore': 0.25,
        'absolute_pct': 0.25,
        'relative_rank': 0.1,
        'delta_pct': 0.4
    }
    
    category_weights = {
        'Quality': 0.25,
        'Valuation': 0.1,
        'Profitability': 0.35,
        'Momentum': 0.3
    }
    
    return factor_dfs, weights, category_weights


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("FACTOR REDUCTION THRESHOLD TESTING")
    print("=" * 70)
    
    # Load data
    factor_dfs, weights, category_weights = load_processed_data()
    
    # Define thresholds to test
    thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\nTesting {len(thresholds_to_test)} threshold values...")
    print(f"Thresholds: {thresholds_to_test}")
    
    # Run threshold testing
    analysis_results = test_factor_reduction_thresholds(
        FactorTransformer,
        original_factor_dfs=factor_dfs,
        metric_weights=weights,
        category_weights=category_weights,
        country_filter='World',
        thresholds_to_test=thresholds_to_test,
        selection_criterion='unique',
        linkage_method='ward'
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)
    print(analysis_results)
    
    # Plot results
    print("\nGenerating visualization...")
    plot_threshold_analysis(analysis_results)
    
    print("\nThreshold testing complete!")

