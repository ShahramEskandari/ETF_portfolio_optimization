"""
Generate Final Portfolio Results

This script generates daily returns for three portfolio strategies:
1. High Risk:   train=50,  test=5   (Aggressive, frequent rebalancing)
2. Low Risk:    train=250, test=10  (Conservative, stable weights)
3. Normal Risk: train=245, test=35  (Balanced approach)

Each strategy runs walk-forward optimization with the specified parameters
and saves the daily returns to a CSV file.

Expected runtime: 1-2 hours per portfolio (3-6 hours total)

Output files:
- dailyReturn_highRisk.csv
- dailyReturn_lowRisk.csv
- dailyReturn_normalRisk.csv

Next step: Move these files to 'plot results/' folder for visualization
"""

import pandas as pd
import numpy as np
from gaOpt2 import *
from makeArrayOfWeights import make_weights
from walkforward import walkForwardOptimization3

print("="*80)
print("GENERATING FINAL PORTFOLIO RESULTS")
print("="*80)

# Load log return data
print("\nLoading return data...")
data_return = pd.read_csv("data/logRetData.csv", index_col=["date"], parse_dates=["date"])
print(f"Data loaded: {len(data_return)} days")

# Generate valid portfolio weight combinations
print("\nGenerating valid portfolio weight combinations...")
allPossibleWeights = make_weights()
print(f"Generated {len(allPossibleWeights)} valid combinations")

# Generate High Risk Portfolio
print("\n" + "="*80)
print("1/3: GENERATING HIGH RISK PORTFOLIO (train=50, test=5)")
print("="*80)
cost_highRisk = walkForwardOptimization3(
    data_return, 
    np.array([50, 5]), 
    allPossibleWeights, 
    create_csv=True, 
    csv_name="dailyReturn_highRisk.csv"
)
print(f"High Risk - Mean Daily Return: {cost_highRisk:.6f}")

# Generate Low Risk Portfolio
print("\n" + "="*80)
print("2/3: GENERATING LOW RISK PORTFOLIO (train=250, test=10)")
print("="*80)
cost_lowRisk = walkForwardOptimization3(
    data_return, 
    np.array([250, 10]), 
    allPossibleWeights, 
    create_csv=True, 
    csv_name="dailyReturn_lowRisk.csv"
)
print(f"Low Risk - Mean Daily Return: {cost_lowRisk:.6f}")

# Generate Normal Risk Portfolio
print("\n" + "="*80)
print("3/3: GENERATING NORMAL RISK PORTFOLIO (train=245, test=35)")
print("="*80)
cost_normalRisk = walkForwardOptimization3(
    data_return, 
    np.array([245, 35]), 
    allPossibleWeights, 
    create_csv=True, 
    csv_name="dailyReturn_normalRisk.csv"
)
print(f"Normal Risk - Mean Daily Return: {cost_normalRisk:.6f}")

# Summary
print("\n" + "="*80)
print("ALL PORTFOLIOS GENERATED SUCCESSFULLY")
print("="*80)
print("\nSummary of Mean Daily Returns:")
print(f"  High Risk:   {cost_highRisk:.6f} ({cost_highRisk*252:.2%} annualized)")
print(f"  Low Risk:    {cost_lowRisk:.6f} ({cost_lowRisk*252:.2%} annualized)")
print(f"  Normal Risk: {cost_normalRisk:.6f} ({cost_normalRisk*252:.2%} annualized)")
print("\nNext Steps:")
print("1. Move the generated CSV files to 'plot results/' folder")
print("2. Open 'plot results/evaluationOfResults.ipynb'")
print("3. Run all cells to generate visualizations and statistics")
print("="*80)
