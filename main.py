"""
Main Optimization Script - Outer GA (GA2)

This script finds the optimal walk-forward analysis parameters:
- train_period: Number of days for training window
- test_period: Number of days for testing window

The GA2 algorithm searches for parameters that maximize the mean daily return
across all walk-forward test periods.

Expected runtime: Several hours (depends on population size and generations)
"""

import pandas as pd
import numpy as np
from gaOpt2 import GA2
from makeArrayOfWeights import make_weights

# Load log return data (output from data/prepare_data.ipynb)
print("Loading return data...")
data_return = pd.read_csv("data/logRetData.csv", index_col=["date"], parse_dates=["date"])
print(f"Data loaded: {len(data_return)} days, {len(data_return.columns)} ETFs")

# Generate all valid portfolio weight combinations (~165,000 combinations)
print("\nGenerating valid portfolio weight combinations...")
allPossibleWeights = make_weights()
print(f"Generated {len(allPossibleWeights)} valid weight combinations")

# Run Outer GA (GA2) to find optimal walk-forward parameters
# Parameters:
#   - population_size=30: Number of parameter sets in each generation
#   - generations=10: Maximum number of generations
#   - data_return: Historical return data
#   - allPossibleWeights: Valid portfolio weights for inner GA
print("\nStarting GA2 optimization...")
print("This will take several hours. Progress will be printed for each generation.\n")

gen_dict = GA2(
    population_size=70, 
    generations=30, 
    data=data_return, 
    possibleWeights=allPossibleWeights
)

# Extract best solution
cost_gen = max(gen_dict["best_cost"])
best_index_in_generation = gen_dict["best_cost"].index(max(gen_dict["best_cost"]))
bestSolution = gen_dict["best_param"][best_index_in_generation]

# Print results
print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)
print(f"Best Parameters: [train_period={int(bestSolution[0])}, test_period={int(bestSolution[1])}]")
print(f"Best Fitness (Mean Daily Return): {cost_gen:.6f}")
print(f"Annualized Return (approx): {cost_gen * 252:.2%}")
print("="*80)
