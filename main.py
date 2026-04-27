"""
Main Optimization Script - Complete Portfolio Optimization Pipeline

This script:
1. Asks user for risk and return coefficients for three portfolios (High Risk, Balanced, Low Risk)
2. Runs GA2 optimization for each portfolio to find optimal walk-forward parameters
3. Saves optimal parameters to a file
4. Generates final CSV files with daily returns for each portfolio

Expected runtime: Several hours per portfolio (depends on population size and generations)
"""

import pandas as pd
import numpy as np
from gaOpt2 import GA2
from makeArrayOfWeights import make_weights
from walkforward import walkForwardOptimization3
from generate_final_results import generate_final_results
import json
import os

def get_user_coefficients():
    """Get risk and return coefficients from user for three portfolios"""
    portfolios = {
        "high_risk": {"name": "High Risk", "default_return": 0.9, "default_risk": 0.1},
        "balanced": {"name": "Balanced", "default_return": 0.7, "default_risk": 0.3},
        "low_risk": {"name": "Low Risk", "default_return": 0.5, "default_risk": 0.5}
    }
    
    coefficients = {}
    
    print("="*80)
    print("Enter Risk and Return Coefficients for Three Portfolios")
    print("="*80)
    print("\nPlease enter coefficients (their sum must equal 1):\n")
    
    for key, info in portfolios.items():
        print(f"\n{info['name']}:")
        while True:
            try:
                weight_return = float(input(f"  Return coefficient (default: {info['default_return']}): ") or info['default_return'])
                weight_risk = float(input(f"  Risk coefficient (default: {info['default_risk']}): ") or info['default_risk'])
                
                if abs(weight_return + weight_risk - 1.0) > 0.01:
                    print(f"  Warning: Sum of coefficients must equal 1. (Current sum: {weight_return + weight_risk:.2f})")
                    continue
                
                coefficients[key] = {
                    "weight_return": weight_return,
                    "weight_risk": weight_risk,
                    "name": info['name']
                }
                print(f"  Coefficients saved: Return={weight_return:.2f}, Risk={weight_risk:.2f}\n")
                break
            except ValueError:
                print("  Please enter a valid number.")
    
    return coefficients

def save_optimal_parameters(optimal_params, output_dir="finalOutputs"):
    """Save optimal parameters to a JSON file"""
    filename = os.path.join(output_dir, "optimal_parameters.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(optimal_params, f, indent=4, ensure_ascii=False)
    print(f"\nOptimal parameters saved to '{filename}'.\n")

def load_optimal_parameters():
    """Load optimal parameters from JSON file if exists"""
    filename = "optimal_parameters.json"
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# Load close price data and calculate log returns
print("Loading close price data...")
close_data = pd.read_csv("data/closeData.csv", index_col=["date"], parse_dates=["date"])
print(f"Close data loaded: {len(close_data)} days, {len(close_data.columns)} ETFs")

# Calculate log returns
print("Calculating log returns...")
log_returns = np.log(close_data / close_data.shift(1)).dropna()
# Rename columns from 'close_*' to 'ret_*'
log_returns.columns = [col.replace('close_', 'ret_') for col in log_returns.columns]

# Use log returns as data_return for optimization
data_return = log_returns
print(f"Log returns calculated: {len(data_return)} days, {len(data_return.columns)} ETFs")

# separation of in sample and out sample data
def time_series_split(df, test_size=0.2):
    n = len(df)
    test_n = int(n * test_size)
    train_n = n - test_n
    
    train = df.iloc[:train_n].copy()
    test  = df.iloc[train_n:].copy()
    
    return train, test

inSample_data, outSample_data = time_series_split(data_return, 0.2)

# Create finalOutputs directory if it doesn't exist
output_dir = "finalOutputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\nCreated output directory: '{output_dir}'")

# Generate all valid portfolio weight combinations (~165,000 combinations)
print("\nGenerating valid portfolio weight combinations...")
print("Please wait a few minutes for initial calculations to complete...")
allPossibleWeights = make_weights()
print(f"Generated {len(allPossibleWeights)} valid weight combinations")

# Get coefficients from user
coefficients = get_user_coefficients()

# Dictionary to store optimal parameters for each portfolio
optimal_parameters = {}

# Run GA2 optimization for each portfolio
print("\n" + "="*80)
print("Starting GA2 Optimization")
print("="*80)
print("This process may take several hours...\n")

for portfolio_key, coeffs in coefficients.items():
    print("\n" + "="*80)
    print(f"Optimizing Portfolio: {coeffs['name']}")
    print(f"Coefficients: Return={coeffs['weight_return']:.2f}, Risk={coeffs['weight_risk']:.2f}")
    print("="*80)
    
    # Run GA2 with specific coefficients
    gen_dict = GA2(
        population_size=10, 
        generations=3, 
        data=inSample_data, 
        possibleWeights=allPossibleWeights,
        weight_return=coeffs['weight_return'],
        weight_risk=coeffs['weight_risk']
    )
    
    # Extract best solution
    cost_gen = max(gen_dict["best_cost"])
    best_index_in_generation = gen_dict["best_cost"].index(max(gen_dict["best_cost"]))
    bestSolution = gen_dict["best_param"][best_index_in_generation]
    
    # Store optimal parameters
    optimal_parameters[portfolio_key] = {
        "train_period": int(bestSolution[0]),
        "test_period": int(bestSolution[1]),
        "fitness": float(cost_gen),
        "weight_return": coeffs['weight_return'],
        "weight_risk": coeffs['weight_risk'],
        "name": coeffs['name']
    }
    
    # Print results
    print(f"\nOptimization completed:")
    print(f"  Optimal parameters: train_period={int(bestSolution[0])}, test_period={int(bestSolution[1])}")
    print(f"  Fitness (Mean Daily Return): {cost_gen:.6f}")
    print(f"  Annualized Return (approx): {cost_gen * 252:.2%}")

# Save optimal parameters to file
save_optimal_parameters(optimal_parameters, output_dir)

# Generate final CSV files for each portfolio
print("\n" + "="*80)
print("Generating Final CSV Files")
print("="*80)

csv_mapping = {
    "high_risk": os.path.join(output_dir, "dailyReturn_highRisk.csv"),
    "balanced": os.path.join(output_dir, "dailyReturn_normalRisk.csv"),
    "low_risk": os.path.join(output_dir, "dailyReturn_lowRisk.csv")
}

for portfolio_key, params in optimal_parameters.items():
    csv_name = csv_mapping[portfolio_key]
    print(f"\n{'='*80}")
    print(f"Generating CSV file for: {params['name']}")
    print(f"Parameters: train={params['train_period']}, test={params['test_period']}")
    print(f"Coefficients: Return={params['weight_return']:.2f}, Risk={params['weight_risk']:.2f}")
    print("="*80)
    
    cost = walkForwardOptimization3(
        pd.concat([inSample_data.tail(params['train_period']), outSample_data], axis=0), 
        np.array([params['train_period'], params['test_period']]), 
        allPossibleWeights, 
        create_csv=True, 
        csv_name=csv_name,
        weight_return=params['weight_return'],
        weight_risk=params['weight_risk']
    )
    
    print(f"CSV file '{csv_name}' created.")
    print(f"  Mean Daily Return: {cost:.6f} ({cost*252:.2%} annualized)")

# Generate final results (metrics, charts, statistical tests)
print("\n" + "="*80)
print("Generating Final Evaluation Results")
print("="*80)
print("This will generate performance metrics, charts, and statistical test results...\n")
generate_final_results(insample_returns=inSample_data, outsample_returns=outSample_data, output_dir=output_dir)

# Final summary
print("\n" + "="*80)
print("All Processes Completed Successfully")
print("="*80)
print("\nResults Summary:")
for portfolio_key, params in optimal_parameters.items():
    print(f"\n{params['name']}:")
    print(f"  Optimal parameters: train={params['train_period']}, test={params['test_period']}")
    print(f"  Mean Daily Return: {params['fitness']:.6f} ({params['fitness']*252:.2%} annualized)")
    print(f"  CSV file: {csv_mapping[portfolio_key]}")

print("\n" + "="*80)
print("All Output Files:")
print(f"All files are saved in the '{output_dir}/' folder:")
print("  - optimal_parameters.json")
print("  - dailyReturn_highRisk.csv")
print("  - dailyReturn_normalRisk.csv")
print("  - dailyReturn_lowRisk.csv")
print("  - Results_metrics.csv")
print("  - Statistical_Tests_Results.csv")
print("  - Cum_highRisk.png")
print("  - Cum_lowRisk.png")
print("  - Cum_normalRisk.png")
print("  - Cum_portfolioS.png")
print("  - DD_normalRisk.png")
print("  - DD_portfolioS.png")
print("="*80)
