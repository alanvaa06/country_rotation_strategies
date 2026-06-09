# Country Rotation Strategy

A quantitative factor-based country equity rotation strategy for systematic investment decisions.

**Author:** Alan Vazquez, CFA  
**Last Updated:** November 2025

---

## Overview

This project implements a comprehensive framework for building, testing, and backtesting country rotation strategies using financial and economic factors. The strategy scores countries based on multi-factor models across **Quality**, **Valuation**, **Profitability**, and **Momentum** categories, then constructs portfolios with configurable selection and weighting schemes.

---

## Project Structure

```
country_rotation/
├── Inputs/                          # Raw financial/economic data (Excel files)
├── ProcessedInputs/                 # Processed and derived metrics
├── Normalized_Scores/               # Factor model output scores
├── Backtest_Results/                # Backtesting results
├── outputs/                         # Analysis outputs
│   ├── backtest_results/
│   ├── ic_analysis/
│   ├── normalized_scores/
│   ├── plots/
│   └── processed_inputs/
├── _archive/                        # Archived/backup files
│
├── ProcessData.py                   # Data loading and processing
├── FactorTransformer.py             # Factor transformation and scoring
├── FactorTesting.py                 # Multi-scenario factor testing
├── backtest.py                      # Backtest engine
├── backtesting_tests.py             # Systematic backtesting framework
├── strategy.py                      # Main strategy execution script
├── test_normalized_scores.py        # IC analysis for normalized scores
├── Threshold_Testing.py             # Factor redundancy threshold analysis
│
├── Classification.xlsx              # Country classification data
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project configuration
└── environment.yml                  # Conda environment
```

---

## Core Modules

### 1. `ProcessData.py`

**Purpose:** Load, process, and transform raw financial data from Excel files.

**Key Features:**
- Reads all Excel files from the `Inputs/` folder
- Loads country classification data (region, segment, type)
- Removes weekends from time series
- Slices data by date and filters countries
- Creates regional aggregations (DM, EM, Asia, Europe, LatAm, World)
- Computes derived metrics:
  - **Yield metrics**: Earnings Yield, Cash Flow Yield, Dividend Yield
  - **Spread metrics**: Yields vs 10-Year Bonds
  - **Growth metrics**: Consensus Sales/EBITDA/Earnings/Cash Flow Growth
  - **Margin metrics**: EBIT, EBITDA, Net Margin
  - **Rolling statistics**: Rolling Earnings, Volatility, Cumulative Flows

**Usage:**
```python
from ProcessData import ProcessData

processor = ProcessData(
    inputs_folder='Inputs',
    classification_file='Classification.xlsx',
    target_date='2010-01-01',
    columns_to_drop=['Saudi Arabia']
)

processed_data, regions_dict, classification = processor.run_full_pipeline(export_data=True)
```

---

### 2. `FactorTransformer.py`

**Purpose:** Transform raw factor data into standardized, comparable scores.

**Key Features:**
- Calculates four metric types per factor:
  - **Z-Score**: Expanding z-score converted to percentile (historical comparison)
  - **Absolute Percentile**: Historical percentile rank for each country
  - **Relative Rank**: Cross-sectional rank at each date
  - **Delta Percentile**: Percentile of 63-day percent changes (momentum)
- Applies factor directionality (higher/lower is better)
- Aggregates factors by category (Valuation, Quality, Profitability, Momentum)
- Calculates composite scores with customizable weights
- Normalizes scores cross-sectionally (0-1 range)
- Analyzes factor redundancy using hierarchical clustering

**Factor Categories:**
| Category | Example Factors |
|----------|-----------------|
| Valuation | P/E, P/B, EV/EBITDA, Earnings Yield Spreads |
| Quality | Debt/Equity, Net Debt/EBITDA, Cash Flow |
| Profitability | ROE, EBIT Margin, Consensus Growth |
| Momentum | Rolling Earnings, Cumulative Flows, Price |

**Usage:**
```python
from FactorTransformer import FactorTransformer

# Initialize with country filter
ft = FactorTransformer(country_filter='World')

# Transform all factors
factor_results = ft.transform_all(factor_dfs)

# Calculate weighted average scores
metric_weights = {'zscore': 0.25, 'absolute_pct': 0.25, 'relative_rank': 0.25, 'delta_pct': 0.25}
weighted_scores = ft.calculate_weighted_average(factor_results, metric_weights)

# Aggregate by category
category_scores, country_scores = ft.aggregate_by_category(weighted_scores)

# Calculate composite scores
category_weights = {'Quality': 0.25, 'Valuation': 0.25, 'Profitability': 0.25, 'Momentum': 0.25}
composite_scores, contributions = ft.calculate_composite_score(category_weights)

# Normalize scores
normalized_scores, rebased_contrib_country, rebased_contrib_factor = ft.normalize_and_rebase_contributions()
```

---

### 3. `FactorTesting.py`

**Purpose:** Test multiple factor weight scenarios across different markets.

**Key Features:**
- Tests multiple metric weight scenarios (Equal, Z-Score Heavy, Relative Focus, etc.)
- Tests multiple category weight scenarios (Balanced, Value-Quality, Growth-Momentum, etc.)
- Runs tests across multiple markets (World, DM, EM, Asia, Europe, LatAm)
- Exports normalized scores for each scenario

**Scenarios Tested:**
- **Metric Weights:** 5 scenarios (Equal, ZScore Heavy, Relative Focus, Momentum Heavy, Historical Focus)
- **Category Weights:** 5 scenarios (Balanced, Value-Quality, Growth-Momentum, Quality Heavy, Valuation Heavy)
- **Markets:** 6 regions (World, DM, EM, Asia, Europe, LatAm)

**Output:** 150 normalized score files (5 × 5 × 6)

---

### 4. `backtest.py`

**Purpose:** Core backtesting engine for country rotation strategies.

**Key Features:**
- **Selection Methods:**
  - **Absolute**: Select countries with scores above a threshold
  - **Relative**: Select top N countries by score change
- **Weighting Methods:**
  - **Equal**: 1/N weighting
  - **Risk Parity**: Inverse variance weighting
- **Benchmark Blending**: Configurable benchmark weight (0-100%)
- **Transaction Costs**: Turnover-based cost calculation
- **Performance Metrics:**
  - Annualized Return/Volatility/Sharpe Ratio
  - Max Drawdown, Win Rate
  - Beta, Up/Down Capture
  - Tracking Error, Information Ratio

**Analysis Methods:**
- `plot_cumulative_returns()` - Portfolio vs Benchmark performance
- `portfolio_turnover_analysis()` - Turnover and transaction cost analysis
- `performance_attribution_analysis()` - Country contribution analysis
- `IC_analysis()` - Information Coefficient analysis
- `plot_weights_over_time()` - Portfolio composition evolution

**Usage:**
```python
from backtest import Backtest

bt = Backtest(
    normalized_score=normalized_score,
    prices=prices,
    selection_criteria='relative',
    relative_selection_score=5,
    weighting_method='Equal',
    bmk='World',
    bmk_weight=0.50,
    periodicity=63,
    transaction_cost_bps=2.0
)

results = bt.run_backtest()
summary = bt.get_performance_summary()
bt.plot_cumulative_returns()
```

---

### 5. `backtesting_tests.py`

**Purpose:** Systematic framework for running parameter grid backtests.

**Key Features:**
- Automated parameter grid generation
- Runs backtests across multiple normalized score files
- Extracts benchmark from filename automatically
- Aggregates results into structured DataFrame
- Exports comprehensive results to Excel

**Parameters Tested:**
- Selection criteria (absolute/relative)
- Absolute selection thresholds (0.60 - 0.90)
- Weighting methods (Equal, Risk Parity)
- Benchmark weights (0% - 50%)
- Periodicities (5, 10, 21, 63 days)

**Usage:**
```python
from backtesting_tests import BacktestingTests

bt_tests = BacktestingTests()
bt_tests.load_price_data()

# Get score files filtered by IC
score_files = bt_tests.get_normalized_score_files()

# Create parameter grid
params = {
    'selection_criteria': ['absolute', 'relative'],
    'weighting_method': ['Equal', 'Risk_Parity'],
    'bmk_weight': [0.0, 0.3, 0.5],
    'periodicity': [5, 21, 63]
}
grid = bt_tests.create_parameter_grid(params)

# Run tests
results = bt_tests.run_multi_score_tests(score_files, grid)
bt_tests.save_results('results.xlsx')
```

---

### 6. `strategy.py`

**Purpose:** Main script for running a single backtest with specific parameters.

**Usage:**
```python
python strategy.py
```

This script:
1. Loads processed prices and normalized scores
2. Configures backtest parameters
3. Runs the backtest
4. Displays performance summary and visualizations

---

### 7. `test_normalized_scores.py`

**Purpose:** Calculate Information Coefficient (IC) statistics for normalized score methods.

**Key Features:**
- Tests IC across multiple periodicities (5, 10, 21, 63 days)
- Tests both Absolute and Relative signal methodologies
- Calculates comprehensive IC statistics:
  - Mean/Median/Std IC
  - IC t-statistic
  - ICIR (IC Information Ratio)
  - Hit Rate

**Methodology:**
- **Absolute Method**: Tests if score levels predict subsequent returns
- **Relative Method**: Tests if score changes (momentum) predict returns

**Output:** `outputs/ic_analysis/ic_analysis_results.xlsx`

---

### 8. `Threshold_Testing.py`

**Purpose:** Analyze factor redundancy and optimal threshold for factor reduction.

**Key Features:**
- Uses hierarchical clustering on factor correlations
- Tests multiple distance thresholds
- Measures impact on composite score rank correlation
- Helps identify optimal number of non-redundant factors

**Visualization:** Plots threshold vs. number of factors and correlation

---

## Data Files

### Input Data (`Inputs/`)

| File | Description |
|------|-------------|
| `Price.xlsx` | Country equity index prices |
| `PE.xlsx`, `Fwd_PE.xlsx` | P/E ratios (TTM and Forward) |
| `PB.xlsx`, `PS.xlsx` | P/B and P/S ratios |
| `EV_EBITDA.xlsx`, `Fwd_EV_EBITDA.xlsx` | EV/EBITDA ratios |
| `DVD.xlsx`, `Fwd_DVD.xlsx` | Dividend yields |
| `ROE.xlsx`, `Fwd_ROE.xlsx` | Return on Equity |
| `Revenue.xlsx`, `EBITDA.xlsx` | Revenue and EBITDA |
| `Debt.xlsx`, `Equity.xlsx` | Balance sheet items |
| `Ten_Year.xlsx` | 10-Year Government Bond Yields |
| `Flows.xlsx` | ETF flows |
| `GDP.xlsx`, `M2.xlsx` | Macro indicators |

### Classification (`Classification.xlsx`)

Contains country metadata:
- **Segment**: DM (Developed Markets) or EM (Emerging Markets)
- **Region**: Europe, Asia, LatAm
- **Type**: Country classification type

---

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using conda

```bash
conda env create -f environment.yml
conda activate country-rotation
```

### Dependencies

- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.9.0` - Statistical functions
- `matplotlib>=3.5.0` - Visualization
- `openpyxl>=3.0.0` - Excel file handling

---

## Quick Start

### 1. Process Raw Data

```python
from ProcessData import ProcessData

processor = ProcessData()
processed_data, regions_dict, classification = processor.run_full_pipeline()
```

### 2. Generate Factor Scores

```python
from FactorTransformer import FactorTransformer

ft = FactorTransformer(country_filter='World')
# ... run transformation pipeline
```

### 3. Run Backtest

```python
from backtest import Backtest

bt = Backtest(normalized_score=scores, prices=prices, ...)
results = bt.run_backtest()
```

---

## Workflow

```
┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐
│   Inputs/   │───▶│  ProcessData.py  │───▶│ ProcessedInputs/  │
│  (Raw Data) │    │                  │    │ (Derived Metrics) │
└─────────────┘    └──────────────────┘    └───────────────────┘
                                                     │
                                                     ▼
┌─────────────────────┐    ┌────────────────────────────────────┐
│ Normalized_Scores/  │◀───│      FactorTransformer.py         │
│  (Factor Scores)    │    │      FactorTesting.py             │
└─────────────────────┘    └────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐    ┌────────────────────────────────────┐
│ Backtest_Results/   │◀───│        backtest.py                │
│  (Performance)      │    │        backtesting_tests.py       │
└─────────────────────┘    └────────────────────────────────────┘
```

---

## Key Concepts

### Factor Scoring

1. **Raw Factor** → Transform into 4 metrics (zscore, absolute_pct, relative_rank, delta_pct)
2. **Metrics** → Weighted average to get single factor score
3. **Factor Scores** → Aggregate by category (Quality, Valuation, Profitability, Momentum)
4. **Category Scores** → Weighted composite score
5. **Composite** → Cross-sectional normalization (0-1)

### Selection Criteria

- **Absolute**: Select all countries with composite score > threshold
- **Relative**: Select top N countries by score change

### Weighting Methods

- **Equal**: All selected countries get equal weight
- **Risk Parity**: Weight inversely proportional to variance

---

## License

MIT License

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
