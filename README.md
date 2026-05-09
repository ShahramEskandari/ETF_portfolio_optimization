# ETF Portfolio Optimization using Nested Genetic Algorithms

A portfolio optimization system that uses **nested genetic algorithms** and **walk-forward analysis** to:

* Optimize ETF portfolio weights (inner GA, GA1)
* Optimize walk-forward train/test window lengths (outer GA, GA2)
* Evaluate **two risk profiles** (High Risk, Low Risk) against **benchmark indices**, a **Markowitz (max-Sharpe)** baseline, and an **equal-weight** baseline

Log returns are built from `data/closeData.csv`. **GA2 is fitted only on the in-sample slice** (first 80% of trading days by default). **Out-of-sample** is the remaining 20%; final evaluation merges GA portfolios, benchmarks, and baselines on overlapping dates.

Running `python main.py` drives the full pipeline and writes outputs under `finalOutputs/`.

---

## Table of Contents

- [Overview](#overview)
- [Project architecture](#project-architecture)
- [Installation](#installation)
- [Usage workflow](#usage-workflow)
- [File descriptions](#file-descriptions)
- [Algorithm details](#algorithm-details)
- [Configuration](#configuration)
- [Output files](#output-files)
- [Notes and troubleshooting](#notes-and-troubleshooting)

---

## Overview

**Nested GA:**

1. **Outer GA (GA2, `gaOpt2.py`)** – optimizes walk-forward parameters:
   - `train_period`: training window length (rows / trading days)
   - `test_period`: test window length  
   Constraint: `train_period >= test_period`; both are drawn from `{10, 20, …, 200}`.
2. **Inner GA (GA1, `gaOpt1.py`)** – optimizes weights for an **8-ETF** long-only portfolio under the same asset-class floors as `makeArrayOfWeights.is_valid`.

**Data split (`main.py`):**
- `time_series_split(data_return, test_size=0.2)` → **in-sample** ≈ first 80% of rows, **out-of-sample** ≈ last 20%.
- **GA2** runs on `inSample_data` only.
- After optimal `(train_period, test_period)` are found, **daily return CSVs** are built by `walkForwardOptimization3` on: `pd.concat([inSample_data.tail(train_period), outSample_data])`  
so the last in-sample tail plus full out-of-sample form the series that walk-forward windows slide over.

**Final evaluation (`generate_final_results.py`):**
- Loads the two GA portfolio CSVs, `data/logRetBenchETF.csv` (`agas`, `close_overal`), and test-period series from:
  - `markowitz_train_test_project` → `markowitz_test_returns.csv`
  - `equal_weight_train_test_project` → `equal_weight_test_returns.csv`  
  (both trained on `inSample_data`, tested on `outSample_data`, matching the 80/20 split)
- **Note:** As of May 7, 2026, the Markowitz baseline is commented out in evaluations within `generate_final_results.py` but can be re-enabled.
- Metrics, plots, and statistical tests use the **inner join** of all these series on `Date`.


## Installation

**Prerequisites:** Python 3.8+, `pip`.

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (CMD)
.\venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

Main libraries: numpy, pandas, scipy (tests + Markowitz optimizer), matplotlib, seaborn; optional pyarrow for pandas I/O.

Usage workflow
1. Input data
data/closeData.csv

Index or column: Date (parsed as dates).

Columns include (names matter for class constraints):
close_afran, close_yaghoot, close_goldMofid, close_nahal, close_sahar, close_agas, close_sarv, close_atlas.

Log returns in code: log(close / close.shift(1)); columns become ret_*.

data/logRetBenchETF.csv

Must include Date (parsed with format='%m/%d/%Y'), plus agas and close_overal log-return columns.

You do not need pre-built logRetData.csv; returns are computed in main.py and evaluation code.

2. Run the pipeline
From the project root:
python main.py

Interactive coefficients (get_user_coefficients):

- Two portfolios: High Risk, Low Risk.

- Defaults if you press Enter:

  - High Risk: return 0.8, risk 0.2

  - Low Risk: return 0.2, risk 0.8

- Coefficients must sum to 1.0 within 0.01 (main.py).

Steps executed:

1- Load closes, compute log returns, 80/20 time split.

2- make_weights() – enumerate valid 8-asset weights (step 0.1); on the order of ~165,000 valid vectors (runtime: minutes).

3- For each profile: GA2(population_size=70, generations=40, data=inSample_data, …) → best (train_period, test_period) and fitness (mean walk-forward test daily return on in-sample).

4- Save finalOutputs/optimal_parameters.json.

5- For each profile: walkForwardOptimization3 on concat(inSample_data.tail(train_period), outSample_data) with create_csv=True → daily return CSVs.

6- generate_final_results(insample_returns=inSample_data, outsample_returns=outSample_data) → baselines, merge, metrics, plots, tests.

Runtime: Highly dependent on data length and GA settings; outer GA with population 70 × generations 40 and inner GA 100 × 50 per walk-forward window can take many hours for two profiles. Consider smaller population_size / generations for experiments.

File descriptions:
  main.py
    get_user_coefficients() – interactive return/risk weights for two profiles.

    save_optimal_parameters() / load_optimal_parameters() – JSON I/O.

    time_series_split – 80% in-sample / 20% out-of-sample.

    Orchestrates weights → GA2 × 2 → CSVs → generate_final_results.

  gaOpt1.py (inner GA)
    generate_gene() – random feasible weights (rounded to 3 decimals, sum 1, class mins).

    GA(population_size, generations, data, possibleWeights, weight_return, weight_risk) – selection (top 75%), crossover, mutation, fitness cache.

    Mutation rates: initial_mutation_rate=0.05, high_mutation_rate=0.1.
    From generation 15 onward: if the best fitness stalls over recent generations, mutation switches to high; if it remains flat under high mutation, the loop breaks early.

  gaOpt2.py (outer GA)
    Chromosome: [train_period, test_period] with train_period >= test_period.

    GA2 uses walkForwardOptimization3 as fitness (mean test-window daily returns), same adaptive mutation / early-stop pattern as GA1 from generation 15.

  walkforward.py
    walk_forward_split – non-overlapping test blocks; advance by test_period.

    walkForwardOptimization3 – for each split: GA(100, 50, train_df, allPossibleWeights, weight_return, weight_risk) , apply best weights to test rows, mean of all test daily returns = fitness; optional CSV of concatenated test returns (Date, Return).

    walkForwardOptimization / walkForwardOptimization2 – older experiments; not used by main.py.

  fitness_ga1.py
    findMinMax over all possibleWeights for scaling.

    fitness_function: fitness = weight_return * ret_scale - weight_risk * risk_scale with ret_scale, risk_scale in [0, 1] from min/max mean return and std across the discrete weight set.

    calc_returnOfPoints – daily portfolio log returns for a weight vector.

  makeArrayOfWeights.py
    make_weights() – Cartesian product of {0.0, 0.1, …, 1.0}^8 filtered by is_valid.

    is_valid: sum ≈ 1, each class ≥ 12.5%, non-negative.

  markowitz_train_test.py
    markowitz_train_test_project(train_returns, test_returns, …) – SLSQP max Sharpe on train, long-only, optional same class constraints as GA; writes test log returns to finalOutputs/markowitz_test_returns.csv when requested.

  equal_weight_train_test.py
    equal_weight_train_test_project – fixed 1/N weights; test log returns to finalOutputs/equal_weight_test_returns.csv when requested.

  generate_final_results.py
    generate_final_results(insample_returns, outsample_returns, output_dir=..., benchmark_file=...) – runs both baselines, merges all series, calculate_metrics, plot_cumulative_returns, plot_drawdowns, perform_statistical_tests (Shapiro, Levene across portfolios, benchmarks, equalweight where columns exist).


Algorithm details:
Nested structure
GA2 (outer, on in-sample data)
  chromosome: [train_period, test_period]
  population: 70, max generations: 40 (early stop possible)
  fitness: mean daily return over all walk-forward test segments
    └── for each segment:
          GA1 (inner): population 100, max generations 50 (early stop possible)
          fitness: scaled return/risk (user return & risk coefficients)
          apply best weights to test segment → daily returns

Portfolio constraints:
  - Weights sum to 1, all ≥ 0.

  - Equity (indices 0–1): sum ≥ 12.5%.

  - Gold (2–4): sum ≥ 12.5%.

  - Fixed income (5–7): sum ≥ 12.5%.

ETF mapping matches close_* column names in data/closeData.csv.

Outer GA fitness
Mean of concatenated test-window daily log returns from walkForwardOptimization3 (maximization).

Configuration:
  GA sizes (current defaults):
    Location	              Setting	        Value
    main.py	                GA2	            population_size=70, generations=40
    walkforward.py	        GA1             inside walkForwardOptimization3	GA(100, 50, …)
    gaOpt1.py / gaOpt2.py	  Mutation	      0.05 → 0.1 when stalled (from gen ≥ 15)
  Train/test split:
    In main.py, change test_size in time_series_split(data_return, 0.2).
    generate_final_results must receive the same inSample_data / outSample_data so Markowitz and equal-weight baselines align with GA out-of-sample dates.

  Walk-forward search space:
    gaOpt2.generate_gene: test_period ∈ range(10, 201, 10), train_period ∈ range(test_period, 201, 10).

Constraints and weight grid:
  Edit makeArrayOfWeights.is_valid (floors, sum tolerance) or step_values (default 0.1) — larger steps reduce count and memory.

Output files:
  All under finalOutputs/ after a successful main.py run:

  Data

  File	                              Description
  optimal_parameters.json	            Per-profile train_period, test_period, fitness, coefficients
  dailyReturn_highRisk.csv	          GA high-risk test-window returns (walk-forward export)
  dailyReturn_lowRisk.csv	            GA low-risk test-window returns
  markowitz_test_returns.csv	        Markowitz baseline, out-of-sample log returns (if enabled)
  equal_weight_test_returns.csv	      Equal-weight baseline, out-of-sample log returns
  Results_metrics.csv	                Mean log return, std, Sharpe, cumulative %, MDD per column
  Statistical_Tests_Results.csv	      Shapiro, Levene results
  Charts

  File	                              Description
  Cum_highRisk.png, Cum_lowRisk.png	  Cumulative wealth (exp cum log return) for each GA profile
  Cum_portfolioS.png	                Both GA profiles together
  Cum_portfolioS-benchmarkS.png	      GA profiles + agas, close_overal, equalweight
  DD_portfolioS.png	                  Drawdowns: both GA profiles
  DD_portfolioS-benchmarkS.png	      Drawdowns: all of the above

  Merged analytics use the intersection of dates across loaded series; ensure benchmarks and ETF data cover the out-of-sample window.

Notes and troubleshooting:
  1- Stochastic runs – GAs use randomness; set numpy / random seeds for reproducibility if needed.

  2 Missing files – generate_final_results warns if a portfolio CSV is missing; merge may fail or drop columns if benchmarks/baselines cannot align.

  3- ModuleNotFoundError (e.g. scipy) – pip install -r requirements.txt.

  4- FileNotFoundError: data/closeData.csv – add data under data/ with expected columns.

  5- Slow or heavy runs – reduce GA2 settings in main.py and GA1 (100, 50) in walkforward.py; quality may suffer.

  6- Memory – shrinking the weight grid (e.g. step 0.2) reduces make_weights() size.

  7- Markowitz baseline – As of the May 7 commit, the Markowitz comparison is commented in generate_final_results.py. Uncomment if needed.

