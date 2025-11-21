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

#%%
prices=pd.read_excel('ProcessedInputs/Price.xlsx', index_col=0,parse_dates=True)
normalized_score=pd.read_excel('Normalized_Scores/NormalizedScores_World_Scenario_3_Relative_Focus_Scenario_A_Balanced.xlsx', index_col=0,parse_dates=True)

#%%

# Configuration parameters
PERIODICITY = 63  # Rebalancing frequency in days
TRANSACTION_COST_BPS = 2.0  # Transaction cost in basis points

backtest = Backtest(
    normalized_score=normalized_score,
    prices=prices,
    selection_criteria="relative",
    absolute_selection_score=0.75,
    weighting_method="Equal",
    relative_selection_score=5,
    bmk='World',
    bmk_weight=0.0,
    periodicity=PERIODICITY,
    transaction_cost_bps=TRANSACTION_COST_BPS
)

results_1 = backtest.run_backtest()
summary_1 = backtest.get_performance_summary()
#%%
IC=backtest.IC_analysis(plot=True)
#%%
IC['statistics']['num_periods']
#%%

summary_1