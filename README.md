# ETF Portfolio Optimization using Nested Genetic Algorithms

A portfolio optimization system that uses **nested genetic algorithms** and **walk-forward analysis** to:
- Optimize ETF portfolio weights (inner GA)
- Optimize walk-forward train/test window lengths (outer GA)
- Evaluate three risk profiles (High Risk, Balanced, Low Risk) and compare them to benchmark indices

All final outputs (JSON, CSVs, charts, metrics, statistical tests) are produced automatically and saved in the `finalOutputs/` folder by running a single script: `main.py`.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage Workflow](#usage-workflow)
- [File Descriptions](#file-descriptions)
- [Algorithm Details](#algorithm-details)
- [Configuration](#configuration)

---

## Overview

This project implements a **two-level (nested) genetic algorithm**:

1. **Outer GA (GA2)** – optimizes walk-forward parameters:
   - `train_period`: length of the training window (days)
   - `test_period`: length of the test window (days)

2. **Inner GA (GA1)** – optimizes weights for an 8-ETF portfolio under allocation constraints.

The system:
- Builds log returns from raw close prices (`data/closeData.csv`)
- Searches for optimal walk-forward parameters for 3 risk profiles (user-defined coefficients)
- Generates daily portfolio returns for each profile
- Computes performance metrics, charts, and statistical tests against benchmarks

All of this is orchestrated from `main.py` in a single execution.

---

## Project Architecture

```text
ETF_portfolio_optimization/
│
├── data/                               # Data directory
│   ├── closeData.csv                   # Close prices (REQUIRED INPUT)
│   └── logRetBenchETF.csv              # Benchmark ETF log returns (REQUIRED INPUT)
│
├── Core Algorithm Files
│   ├── main.py                         # Main pipeline: GA2 + GA1 + final outputs
│   ├── gaOpt1.py                       # Inner GA: portfolio weight optimization
│   ├── gaOpt2.py                       # Outer GA: walk-forward parameter optimization
│   ├── fitness_ga1.py                  # Fitness functions for portfolio evaluation
│   ├── walkforward.py                  # Walk-forward analysis implementation
│   └── makeArrayOfWeights.py           # Generation of valid portfolio weight combinations
│
├── Evaluation & Results
│   └── generate_final_results.py       # Pure-Python evaluation (metrics, charts, tests)
│
├── finalOutputs/                       # All final outputs created by main.py
│   ├── optimal_parameters.json
│   ├── dailyReturn_highRisk.csv
│   ├── dailyReturn_lowRisk.csv
│   ├── dailyReturn_normalRisk.csv
│   ├── Results_metrics.csv
│   ├── Statistical_Tests_Results.csv
│   ├── Cum_highRisk.png
│   ├── Cum_lowRisk.png
│   ├── Cum_normalRisk.png
│   ├── Cum_portfolioS.png
│   ├── DD_normalRisk.png
│   └── DD_portfolioS.png
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### Setup Steps

1. **Create and activate a virtual environment**

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (CMD)
.\venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy>=1.24.0` – numerical computations
- `pandas>=2.0.0` – data manipulation
- `scipy>=1.11.0` – statistical tests (used in `generate_final_results.py`)
- `matplotlib>=3.7.0`, `seaborn>=0.12.0` – plotting
- `pyarrow>=12.0.0` – optional, speeds up some pandas operations

---

## Usage Workflow

The project is designed so that **one command** runs the entire optimization and evaluation pipeline.

### STEP 1 – Prepare Input Data

**Required files:**

1. **Close prices**: `data/closeData.csv`
   - Format: CSV with `date` as index column
   - Columns: `date`, `close_afran`, `close_yaghoot`, `close_goldMofid`, `close_nahal`, `close_sahar`, `close_agas`, `close_sarv`, `close_atlas`
   - Date format: YYYY-MM-DD
   - Example:
     ```csv
     date,close_afran,close_yaghoot,close_goldMofid,close_nahal,close_sahar,close_agas,close_sarv,close_atlas
     2021-02-13,12663,11878,31124.36,9865.96,10908.05,69174.34,6360.23,20902.53
     ```

2. **Benchmark returns**: `data/logRetBenchETF.csv`
   - Format: CSV with `date` as index column
   - Required columns: `date`, `agas`, `close_overal`
   - Date format: M/D/YYYY (e.g., `2/15/2021`)
   - Contains log returns for benchmark ETFs

> **Note**: Log returns for the optimization are computed automatically inside `main.py` from `closeData.csv`. You do **not** need to pre-generate `logRetData.csv` or `simpleRetData.csv`.

### STEP 2 – Run Full Optimization & Evaluation

From the project root:

```bash
python main.py
```

**What `main.py` does:**

1. **Data loading & preprocessing**
   - Reads `data/closeData.csv`
   - Computes log returns: `log_returns = log(close / close.shift(1))`
   - Renames columns from `close_*` to `ret_*`
   - Creates `finalOutputs/` directory if it doesn't exist

2. **Portfolio weight space generation**
   - Calls `make_weights()` in `makeArrayOfWeights.py`
   - Generates all valid 8-asset weight combinations (step size: 0.1)
   - Filters by portfolio constraints (~165,000 valid combinations)
   - **Note**: This step takes a few minutes

3. **User-defined risk/return preferences**
   - Prompts user for **return** and **risk** coefficients for three portfolios:
     - **High Risk** (default: return=0.9, risk=0.1)
     - **Balanced** (default: return=0.7, risk=0.3)
     - **Low Risk** (default: return=0.5, risk=0.5)
   - User can press Enter to accept defaults or enter custom values
   - Validation: coefficients must sum to 1.0

4. **Outer GA optimization (GA2) for each profile**
   - For each of the three risk profiles:
     - Runs `GA2` (from `gaOpt2.py`) with profile-specific `(weight_return, weight_risk)`
     - **GA2 parameters**: `population_size=40`, `generations=10`
     - Searches over `[train_period, test_period]`:
       - `train_period ∈ {5, 10, ..., 250}` (steps of 5)
       - `test_period ∈ {5, 10, ..., 250}` (steps of 5)
       - Constraint: `train_period >= test_period`
     - Uses walk-forward analysis (`walkForwardOptimization3`) and inner GA (`GA`) to evaluate each candidate
   - Stores optimal parameters for each profile:
     - `train_period` and `test_period`
     - Best fitness (mean daily return)
     - Risk/return weights used
   - Saves to: `finalOutputs/optimal_parameters.json`

5. **Generate daily return series for each portfolio**
   - For each risk profile:
     - Re-runs `walkForwardOptimization3` using optimal `(train_period, test_period)` and `(weight_return, weight_risk)`
     - **Inner GA parameters** (used in walk-forward): `population_size=40`, `generations=10`
     - Saves daily portfolio returns to:
       - `finalOutputs/dailyReturn_highRisk.csv`
       - `finalOutputs/dailyReturn_normalRisk.csv` (note: "Balanced" → "normalRisk" in filename)
       - `finalOutputs/dailyReturn_lowRisk.csv`

6. **Generate final evaluation outputs**
   - Calls `generate_final_results(output_dir="finalOutputs", benchmark_file="data/logRetBenchETF.csv")`
   - This script:
     - Loads benchmark returns from `data/logRetBenchETF.csv`
     - Merges portfolio returns with benchmarks
     - Computes performance metrics:
       - Mean Daily Log Return
       - Standard Deviation
       - Sharpe Ratio
       - Cumulative Return %
       - Maximum Drawdown (MDD)
     - Saves metrics to: `finalOutputs/Results_metrics.csv`
     - Generates charts:
       - `finalOutputs/Cum_highRisk.png`
       - `finalOutputs/Cum_lowRisk.png`
       - `finalOutputs/Cum_normalRisk.png`
       - `finalOutputs/Cum_portfolioS.png` (all portfolios comparison)
       - `finalOutputs/DD_normalRisk.png`
       - `finalOutputs/DD_portfolioS.png` (all portfolios drawdown comparison)
     - Runs statistical tests:
       - Shapiro-Wilk normality tests (for each portfolio/benchmark)
       - Levene variance equality tests (portfolio comparisons)
       - Mann-Whitney U tests (portfolio vs benchmark comparisons)
     - Saves test results to: `finalOutputs/Statistical_Tests_Results.csv`

7. **Final summary**
   - Prints summary of:
     - Optimal walk-forward parameters for each risk profile
     - Mean daily returns (and annualized approximations)
     - Complete list of all files written to `finalOutputs/`

**Expected runtime:**
- Weight generation: ~2-5 minutes
- GA2 optimization per portfolio: Several hours (depends on data size and GA parameters)
- Daily return generation: ~1-2 hours per portfolio
- Final evaluation: ~1-2 minutes

**Total**: Several hours to complete all three portfolios (consider running overnight or on a powerful machine).

---

## File Descriptions

### Core Algorithm Files

#### `main.py`
Main entry point for the entire pipeline.

**Functions:**
- `get_user_coefficients()`: Interactive prompt for risk/return coefficients
- `save_optimal_parameters()`: Saves optimal parameters to JSON
- `load_optimal_parameters()`: Loads saved parameters (currently unused but available)

**Workflow:**
1. Loads `data/closeData.csv` and computes log returns
2. Creates `finalOutputs/` directory
3. Generates valid weight combinations
4. Collects user coefficients for three portfolios
5. Runs GA2 optimization for each portfolio
6. Generates daily return CSVs
7. Calls `generate_final_results()` for metrics, charts, and tests

#### `gaOpt1.py` (Inner GA)
Optimizes 8-ETF portfolio weights.

**Key Functions:**
- `generate_gene()`: Creates random valid weight vectors (8 weights summing to 1, satisfying constraints)
- `create_population(population_size)`: Initializes population
- `selection()`: Selects top 75% of population based on fitness
- `crossover()`: Single-point crossover with normalization
- `mutation()`: Random gene mutation with constraint checking
- `GA(population_size, generations, data, possibleWeights, weight_return=0.8, weight_risk=0.2)`: Main GA loop

**Parameters:**
- `initial_mutation_rate = 0.05` (5%)
- `high_mutation_rate = 0.1` (10%)
- Adaptive mutation: switches to high rate if no improvement for 8 consecutive generations
- Early stopping: stops if still no improvement after high mutation period

**Constraints enforced:**
- Sum of weights = 1.0
- Equity ETFs (indices 0-1): sum ≥ 12.5%
- Gold ETFs (indices 2-4): sum ≥ 12.5%
- Fixed Income ETFs (indices 5-7): sum ≥ 12.5%
- All weights ≥ 0 (no short selling)

#### `gaOpt2.py` (Outer GA)
Optimizes `[train_period, test_period]` parameters.

**Key Functions:**
- `generate_gene()`: Creates random `[train_period, test_period]` with `train_period >= test_period`
- `create_population(population_size)`: Initializes population
- `selection()`: Selects top 75% based on fitness (calls `walkForwardOptimization3`)
- `crossover()`: Single-point crossover for period parameters
- `mutation()`: Mutates periods within valid ranges (maintains `train_period >= test_period`)
- `GA2(population_size, generations, data, possibleWeights, weight_return=0.8, weight_risk=0.2)`: Main GA loop

**Parameters:**
- `initial_mutation_rate = 0.05` (5%)
- `high_mutation_rate = 0.1` (10%)
- Same adaptive mutation and early stopping logic as GA1

**Chromosome structure:**
```python
np.array([train_period, test_period])
# train_period ∈ {5, 10, ..., 250}
# test_period ∈ {5, 10, ..., 250}
# Constraint: train_period >= test_period
```

#### `walkforward.py`
Implements walk-forward analysis.

**Key Functions:**
- `walk_forward_split(df, arr_train_test)`: Splits time series into rolling train/test windows
  - Step size = `test_period` (non-overlapping test sets)
  - Returns list of `(train_df, test_df)` tuples

- `walkForwardOptimization3(dataframe, train_test_array, allPossibleWeights, create_csv=False, csv_name=None, weight_return=0.8, weight_risk=0.2)`:
  - Main walk-forward function
  - For each train window:
    - Runs `GA(population_size=40, generations=10, ...)` to find optimal weights
  - For each corresponding test window:
    - Applies optimal weights to compute daily returns
  - Returns: mean daily return across all test periods
  - If `create_csv=True`:
    - Concatenates all test-period daily returns
    - Saves to CSV with `Date` index and `Return` column

**Other functions (legacy/unused):**
- `walkForwardOptimization()`: Uses `GA(70, 40, ...)` with risk-adjusted return metric
- `walkForwardOptimization2()`: Uses `GA(200, 50, ...)` with benchmark threshold

#### `fitness_ga1.py`
Portfolio evaluation functions.

**Key Functions:**
- `fitness_function(ret_data, arr_weights, extermums, weight_return=0.8, weight_risk=0.2)`:
  - Calculates portfolio mean return and standard deviation
  - Scales return and risk to [0, 1] based on extrema
  - Returns weighted combination:
    ```
    fitness = weight_return * return_scaled - weight_risk * risk_scaled
    ```

- `calc_riskAdjRet(ret_data, arr_weights)`: Sharpe-like ratio (mean/std)

- `calc_returnOfPoints(ret_data, arr_weights)`: Returns array of daily portfolio returns

- `findMinMax(data, weights)`: Finds min/max return and risk across all valid weights for normalization

**Default values:**
- `weightReturn = 0.8`
- `weightRisk = 0.2`

#### `makeArrayOfWeights.py`
Generates all valid portfolio weight combinations.

**Functions:**
- `make_weights()`:
  - Generates all 8-asset combinations with step size = 0.1
  - Filters by `is_valid()` constraints
  - Returns ~165,000 valid combinations

- `is_valid(gene)`:
  - Validates: sum = 1.0, asset class minimums, non-negativity

### Evaluation & Results

#### `generate_final_results.py`
Standalone evaluation module (imported by `main.py`).

**Functions:**
- `generate_final_results(output_dir="finalOutputs", benchmark_file="data/logRetBenchETF.csv")`: Main function
- `calculate_metrics(returns)`: Computes performance metrics and drawdowns
- `plot_cumulative_returns(returns, output_dir, portfolios)`: Generates cumulative return charts
- `plot_drawdowns(drawdowns, output_dir, portfolios)`: Generates drawdown charts
- `perform_statistical_tests(returns, output_dir)`: Runs statistical tests and saves results

**Inputs:**
- Portfolio CSVs from `finalOutputs/` (dailyReturn_highRisk.csv, etc.)
- Benchmark returns from `data/logRetBenchETF.csv`

**Outputs:**
- `Results_metrics.csv`: Performance metrics table
- `Statistical_Tests_Results.csv`: Statistical test results
- 6 PNG charts: Cumulative returns and drawdowns

---

## Algorithm Details

### Nested GA Architecture

```
GA2 (Outer)
├─ Chromosome: [train_period, test_period]
├─ Population: 40 individuals
├─ Generations: 10 (max)
├─ Fitness: Mean return from walk-forward analysis
└─ Evaluation:
    └─ Walk-Forward Analysis
        ├─ Split data into train/test windows
        └─ For each window:
            ├─ GA1 (Inner)
            │   ├─ Chromosome: [w1, w2, ..., w8] (8 ETF weights)
            │   ├─ Population: 40 individuals
            │   ├─ Generations: 10 (max)
            │   ├─ Fitness: Scaled return-risk combination
            │   └─ Find optimal weights for train period
            └─ Evaluate optimal weights on test period
```

### Portfolio Constraints

All portfolios must satisfy:

1. **Sum to 1**: `Σ(weights) = 1.0`

2. **Asset Class Minimums**:
   - Equity ETFs (afran + yaghoot): `≥ 12.5%`
   - Gold ETFs (goldMofid + nahal + sahar): `≥ 12.5%`
   - Fixed Income ETFs (agas + sarv + atlas): `≥ 12.5%`

3. **Non-negativity**: `all weights ≥ 0` (no short selling)

**ETF Categories:**
```python
Equity:       [afran, yaghoot]           # Indices 0-1
Gold:         [goldMofid, nahal, sahar]  # Indices 2-4
Fixed Income: [agas, sarv, atlas]        # Indices 5-7
```

### Fitness Function

**Inner GA (Portfolio Weights):**
```python
fitness = weight_return * return_scaled - weight_risk * risk_scaled

where:
  return_scaled = (mean_return - min_return) / (max_return - min_return)
  risk_scaled = (portfolio_std - min_std) / (max_std - min_std)
```

**Outer GA (Walk-Forward Parameters):**
```python
fitness = mean(daily_returns_across_all_test_periods)
```

### Adaptive Mutation & Early Stopping

Both GAs use adaptive mutation rates:
- **Initial**: 0.05 (5% mutation probability)
- **High**: 0.1 (10% mutation probability)

**Trigger Condition:**
- If best fitness doesn't improve for 8 consecutive generations:
  - Switch to high mutation rate (exploration)
  - If still no improvement after high mutation period: stop (converged)

**Selection:**
- Top 75% of population selected for next generation

---

## Configuration

### Modifying GA Parameters

**In `main.py`:**
```python
gen_dict = GA2(
    population_size=40,    # Outer GA population size
    generations=10,         # Outer GA max generations
    ...
)
```

**In `walkforward.py` (line 99):**
```python
gen_dict = GA(40, 10, p[0], allPossibleWeights, weight_return, weight_risk)
#            ↑   ↑
#         pop  gen (Inner GA parameters)
```

**In `gaOpt1.py` and `gaOpt2.py`:**
```python
initial_mutation_rate = 0.05  # 5%
high_mutation_rate = 0.1     # 10%
```

### Modifying Portfolio Constraints

**In `makeArrayOfWeights.py`:**
```python
def is_valid(gene):
    return (
        abs(sum(gene) - 1.0) < 1e-6 and
        sum(gene[0:2]) >= 0.125 and  # Equity minimum
        sum(gene[2:5]) >= 0.125 and  # Gold minimum
        sum(gene[5:8]) >= 0.125 and  # Fixed income minimum
        all(g >= 0 for g in gene)
    )
```

### Modifying Fitness Weights

**In `fitness_ga1.py`:**
```python
weightReturn = 0.8  # Return importance (default)
weightRisk = 0.2    # Risk importance (default)
```

**Note**: These defaults are overridden by user input in `main.py` for each risk profile.

### Modifying Search Space

**In `gaOpt2.py` (line 10, 14):**
```python
test_period = random.choice(range(5, 251, 5))  # 5 to 250, step 5
train_period = random.choice(range(test_period, 251, 5))  # >= test_period
```

---

## Output Files

All outputs are saved in `finalOutputs/`:

### Data Files
- `optimal_parameters.json`: Optimal train/test periods and coefficients for each portfolio
- `dailyReturn_highRisk.csv`: Daily log returns for high-risk portfolio
- `dailyReturn_normalRisk.csv`: Daily log returns for balanced portfolio
- `dailyReturn_lowRisk.csv`: Daily log returns for low-risk portfolio
- `Results_metrics.csv`: Performance metrics (Mean Return, Std Dev, Sharpe Ratio, Cumulative Return %, MDD)
- `Statistical_Tests_Results.csv`: Results of normality, variance, and comparison tests

### Charts
- `Cum_highRisk.png`: Cumulative return chart (high-risk)
- `Cum_lowRisk.png`: Cumulative return chart (low-risk)
- `Cum_normalRisk.png`: Cumulative return chart (balanced)
- `Cum_portfolioS.png`: All portfolios cumulative return comparison
- `DD_normalRisk.png`: Drawdown chart (balanced)
- `DD_portfolioS.png`: All portfolios drawdown comparison

---

## Notes & Best Practices

1. **Computation Time**:
   - Weight generation: ~2-5 minutes
   - GA2 optimization: Several hours per portfolio (depends on data size)
   - Daily return generation: ~1-2 hours per portfolio
   - Consider running overnight or on a powerful machine

2. **Random Seed**:
   - Results may vary between runs (stochastic optimization)
   - Set `np.random.seed()` for reproducibility if needed

3. **Data Requirements**:
   - Minimum 500 days of data recommended
   - Daily frequency required
   - No missing values in return series

4. **Memory Usage**:
   - `make_weights()` generates ~165K combinations (~10MB)
   - Consider reducing step size in `makeArrayOfWeights.py` for memory constraints

5. **Overfitting Prevention**:
   - Walk-forward analysis provides out-of-sample validation
   - Multiple test periods reduce overfitting risk
   - Compare against benchmarks (agas, close_overal) in final evaluation

---

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'scipy'`
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: `FileNotFoundError: data/closeData.csv`
- **Solution**: Ensure `closeData.csv` exists in the `data/` folder with correct format

**Issue**: Long computation time
- **Solution**: Reduce `population_size` or `generations` in `main.py` and `walkforward.py` (may affect solution quality)

**Issue**: Memory error during weight generation
- **Solution**: Modify `makeArrayOfWeights.py` to use larger step size (e.g., 0.2 instead of 0.1)

---

**Last Updated**: January 2025

**Version**: 2.0
