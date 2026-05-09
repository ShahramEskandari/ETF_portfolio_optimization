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
```

### Main libraries

- numpy
- pandas
- scipy (tests + Markowitz optimizer)
- matplotlib
- seaborn
- optional: pyarrow for pandas I/O

---

## Usage Workflow

### 1. Input Data

#### `data/closeData.csv`

- Index or column: `Date` (parsed as dates)
- Columns include:

```text
close_afran
close_yaghoot
close_goldMofid
close_nahal
close_sahar
close_agas
close_sarv
close_atlas
```

Column names matter for class constraints.

Log returns in code:

```python
log(close / close.shift(1))
```

Columns become:

```text
ret_*
```

---

#### `data/logRetBenchETF.csv`

Must include:

- `Date` (parsed with `format='%m/%d/%Y'`)
- `agas`
- `close_overal`

You do not need a pre-built `logRetData.csv`.

Returns are computed dynamically in `main.py`.

---

### 2. Run the Pipeline

From the project root:

```bash
python main.py
```

---

### Interactive Coefficients (`get_user_coefficients`)

Two portfolios:

- High Risk
- Low Risk

Default values if Enter is pressed:

| Portfolio | Return Weight | Risk Weight |
|---|---|---|
| High Risk | 0.8 | 0.2 |
| Low Risk | 0.2 | 0.8 |

Coefficients must sum to `1.0 ± 0.01`.

---

### Pipeline Steps

1. Load closes, compute log returns, perform 80/20 time split.

2. `make_weights()`
   - Enumerates valid 8-asset weights
   - Step size: `0.1`
   - Roughly `~165,000` valid vectors

3. For each profile:

```python
GA2(
    population_size=70,
    generations=40,
    data=inSample_data,
    ...
)
```

Outputs:
- best `(train_period, test_period)`
- fitness score

4. Save:

```text
finalOutputs/optimal_parameters.json
```

5. Run:

```python
walkForwardOptimization3(...)
```

Exports daily return CSVs.

6. Run:

```python
generate_final_results(...)
```

Creates:
- baselines
- merged analytics
- metrics
- plots
- statistical tests

---

### Runtime Notes

Runtime is highly dependent on:
- dataset size
- GA parameters
- hardware

Current nested GA setup may take many hours.

For faster experimentation:
- reduce population size
- reduce generations

---

## File Descriptions

### `main.py`

- `get_user_coefficients()`
- `save_optimal_parameters()`
- `load_optimal_parameters()`
- `time_series_split`

Responsibilities:
- compute weights
- run GA2
- export CSVs
- generate final analytics

---

### `gaOpt1.py` (Inner GA)

Functions:

- `generate_gene()`
- `GA(...)`

Features:
- selection: top 75%
- crossover
- mutation
- fitness cache

Mutation behavior:
- initial mutation: `0.05`
- high mutation: `0.1`

Adaptive logic starts after generation 15.

---

### `gaOpt2.py` (Outer GA)

Chromosome structure:

```text
[train_period, test_period]
```

Constraint:

```text
train_period >= test_period
```

Uses:
- walk-forward optimization
- adaptive mutation
- early stopping

---

### `walkforward.py`

Key functions:

- `walk_forward_split`
- `walkForwardOptimization3`

Workflow:
1. split data
2. run inner GA
3. apply best weights
4. compute test returns
5. export CSVs

Older unused functions:
- `walkForwardOptimization`
- `walkForwardOptimization2`

---

### `fitness_ga1.py`

Functions:

- `findMinMax`
- `fitness_function`
- `calc_returnOfPoints`

Fitness equation:

```text
fitness =
    weight_return * ret_scale
    - weight_risk * risk_scale
```

---

### `makeArrayOfWeights.py`

Functions:

- `make_weights()`
- `is_valid()`

Rules:
- weights sum to 1
- non-negative
- each asset class ≥ 12.5%

---

### `markowitz_train_test.py`

Implements:
- SLSQP maximum Sharpe optimization
- long-only constraints

Exports:

```text
finalOutputs/markowitz_test_returns.csv
```

---

### `equal_weight_train_test.py`

Implements:
- equal-weight benchmark

Exports:

```text
finalOutputs/equal_weight_test_returns.csv
```

---

### `generate_final_results.py`

Main responsibilities:
- merge results
- calculate metrics
- generate plots
- perform statistical tests

Tests include:
- Shapiro
- Levene

---

## Algorithm Details

### Nested Structure

```text
GA2 (outer)
│
├── chromosome: [train_period, test_period]
├── population: 70
├── generations: 40
│
└── For each walk-forward segment:
      GA1 (inner)
      ├── population: 100
      ├── generations: 50
      └── optimize portfolio weights
```

---

### Portfolio Constraints

- weights sum to 1
- all weights ≥ 0

Minimum allocation per class:

| Asset Class | Index Range | Minimum |
|---|---|---|
| Equity | 0–1 | 12.5% |
| Gold | 2–4 | 12.5% |
| Fixed Income | 5–7 | 12.5% |

---

### Outer GA Fitness

Fitness is:

```text
Mean of all walk-forward test daily log returns
```

---

## Configuration

### GA Settings

| File | Setting | Value |
|---|---|---|
| `main.py` | GA2 | population=70, generations=40 |
| `walkforward.py` | GA1 | GA(100, 50) |
| `gaOpt1.py` / `gaOpt2.py` | mutation | 0.05 → 0.1 |

---

### Train/Test Split

In `main.py`:

```python
time_series_split(data_return, 0.2)
```

---

### Walk-Forward Search Space

In `gaOpt2.generate_gene`:

```python
test_period ∈ range(10, 201, 10)
train_period ∈ range(test_period, 201, 10)
```

---

### Weight Grid

Edit:
- `makeArrayOfWeights.is_valid`
- `step_values`

Larger step sizes reduce:
- runtime
- memory usage

---

## Output Files

All outputs are stored in:

```text
finalOutputs/
```

---

### Data Files

| File | Description |
|---|---|
| `optimal_parameters.json` | best parameters per profile |
| `dailyReturn_highRisk.csv` | high-risk GA returns |
| `dailyReturn_lowRisk.csv` | low-risk GA returns |
| `markowitz_test_returns.csv` | Markowitz benchmark |
| `equal_weight_test_returns.csv` | equal-weight benchmark |
| `Results_metrics.csv` | performance metrics |
| `Statistical_Tests_Results.csv` | statistical test outputs |

---

### Charts

| File | Description |
|---|---|
| `Cum_highRisk.png` | cumulative returns |
| `Cum_lowRisk.png` | cumulative returns |
| `Cum_portfolioS.png` | combined portfolios |
| `Cum_portfolioS-benchmarkS.png` | portfolios + benchmarks |
| `DD_portfolioS.png` | drawdowns |
| `DD_portfolioS-benchmarkS.png` | benchmark drawdowns |

---

## Notes and Troubleshooting

1. Stochastic behavior:
   - set random seeds for reproducibility

2. Missing CSV files:
   - merge/alignment may fail

3. Missing dependencies:

```bash
pip install -r requirements.txt
```

4. Missing data file:

```text
data/closeData.csv
```

5. Slow execution:
   - reduce GA sizes

6. High memory usage:
   - increase weight step size

7. Markowitz baseline:
   - currently commented in `generate_final_results.py`