# Country Equity Rotation Strategy

## Overview

This repository contains a quantitative investment framework for a **Country Equity Rotation Strategy**. The project implements a systematic approach to selecting and weighting country equity markets based on multi-factor models. It includes a complete pipeline from raw economic/financial data processing to factor transformation, scoring, and rigorous backtesting.

## Project Structure

The codebase is organized into data processing, score generation, and backtesting modules:

### Core Modules

- **`backtest.py`**: The core backtesting engine. Contains the `Backtest` class which simulates the strategy, handling rebalancing, signal generation, portfolio construction (Equal Weight, Risk Parity), transaction costs, and performance reporting.
- **`backtesting_tests.py`**: A testing framework for running systematic parameter grid searches and analyzing performance across multiple normalized score files.
- **`FactorTesting.py`**: A script for testing different factor combinations, weighting scenarios, and market configurations (World, DM, EM, etc.). It generates normalized scores used by the backtester.
- **`Threshold_Testing.py`**: A specialized module for analyzing factor redundancy. It tests different correlation thresholds to reduce the number of factors while maintaining signal integrity.
- **`strategy.py`**: A streamlined script to run a single backtest configuration and generate immediate visualizations and reports.

### Data Processing

- **`ProcessData.py`**: Handles loading raw Excel data from the `Inputs/` directory, cleaning, alignment, and initial processing.
- **`FactorTransformer.py`**: Transforms raw factors into normalized scores using z-scores, percentile ranks, and other metrics. It handles the aggregation of factors into categories (Valuation, Quality, Momentum, Profitability) and calculates the final composite scores.

## Key Features

### Backtesting Engine (`Backtest`)
- **Selection Criteria**: Supports 'Absolute' (threshold-based) and 'Relative' (top N based on score change/momentum) selection.
- **Weighting Methods**: 
  - `Equal`: 1/N weighting.
  - `Risk_Parity`: Inverse volatility weighting (requires lookback window).
- **Transaction Costs**: Configurable basis points per trade.
- **Performance Analysis**:
  - Comprehensive metrics: Sharpe, Sortino, Drawdown, Beta, Up/Down Capture.
  - **Attribution Analysis**: Holdings-based attribution to identify top contributing countries.
  - **Turnover Analysis**: Tracks trading activity and cost impact.
  - **IC Analysis**: Information Coefficient analysis to validate predictive power of scores.

### Systematic Testing (`BacktestingTests`)
- Automated parameter grid generation.
- Multi-scenario testing across different normalized score files.
- Results aggregation and export to Excel.
- Identification of top-performing configurations.

## Usage

### 1. Generating Scores (`FactorTesting.py`)
Run this script to process raw inputs and generate normalized score files in the `Normalized_Scores/` directory. You can configure:
- Market definitions (World, DM, EM, etc.)
- Factor selection
- Weighting scenarios (e.g., "Value Heavy", "Quality Focused")

```bash
python FactorTesting.py
```

### 2. Running a Single Backtest (`strategy.py`)
For a quick analysis of a specific configuration:
1. Update the `normalized_score` path and parameters in `strategy.py`.
2. Run the script:

```bash
python strategy.py
```

### 3. Systematic Parameter Search (`backtesting_tests.py`)
To test multiple parameters (thresholds, lookbacks, weighting schemes) at once:

```bash
python backtesting_tests.py
```

### 4. Factor Redundancy Analysis (`Threshold_Testing.py`)
To analyze if you can reduce the number of factors without losing information:

```bash
python Threshold_Testing.py
```

## Data Requirements

The system expects Excel files in the `Inputs/` directory covering various fundamental and economic indicators (e.g., `PE.xlsx`, `GDP.xlsx`, `Debt.xlsx`). A `Classification.xlsx` file is used to define regions and markets.

## Outputs

- **`Backtest_Results/`**: Contains detailed Excel reports of backtest runs.
- **`Normalized_Scores/`**: Generated score files used as inputs for backtesting.
- **`ProcessedInputs/`**: Intermediate processed data files.
