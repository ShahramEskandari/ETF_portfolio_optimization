# ETF Portfolio Optimization with Nested Genetic Algorithms

This repository contains the code used in the paper for **multi-asset ETF portfolio optimization** based on a **nested genetic algorithm (GA)** and **walk-forward evaluation**.

The code:

1. Optimizes portfolio weights for an 8-ETF universe (inner GA).
2. Optimizes walk-forward training/testing window lengths (outer GA).
3. Evaluates two risk preference profiles (**High Risk** and **Low Risk**) out of sample.
4. Compares the optimized portfolios with market benchmarks and an equal-weight baseline.
5. Provides optional sensitivity and statistical analysis scripts for robustness checks.

All numerical and graphical results are stored in `finalOutputs/`.

---

## Table of Contents

- [Research summary](#research-summary)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [How to reproduce the main results](#how-to-reproduce-the-main-results)
- [Method overview](#method-overview)
- [Core modules](#core-modules)
- [Optional analysis scripts](#optional-analysis-scripts)
- [Contents of `finalOutputs/`](#contents-of-finaloutputs)
- [Notes for reviewers](#notes-for-reviewers)

---

## Research summary

### Problem

Construct long-only portfolios from eight Iranian ETF assets under realistic allocation constraints, and choose walk-forward window lengths automatically rather than by ad hoc tuning.

### Approach

A **two-level (nested) genetic algorithm**:

| Level | Role | Decision variables |
|---|---|---|
| **Outer GA (GA2)** | Chooses walk-forward window lengths | `(train_period, test_period)` |
| **Inner GA (GA1)** | Chooses portfolio weights on each train window | 8 asset weights summing to 1 |

Risk preference enters the inner fitness as:

```text
fitness = alpha * normalized_return - beta * normalized_risk
where beta = 1 - alpha
```

### Risk profiles used in this repository

`main.py` prompts for coefficients interactively. Pressing Enter uses the **code defaults**:

| Profile | Code default alpha | Code default beta |
|---|---|---|
| High Risk | 0.8 | 0.2 |
| Low Risk | 0.2 | 0.8 |

The **shipped results** in `finalOutputs/optimal_parameters.json` were produced with:

| Profile | alpha (`weight_return`) | beta (`weight_risk`) | Optimal window |
|---|---|---|---|
| High Risk | **0.9** | **0.1** | train=50, test=20 |
| Low Risk | **0.2** | **0.8** | train=90, test=60 |

Please use the JSON file (not only the code defaults) when interpreting the reported tables and charts.

### Evaluation design

- Daily log returns from close prices.
- Chronological split: **80% in-sample** / **20% out-of-sample**.
- GA2 is trained **only on in-sample data**.
- Final GA daily returns are produced by walk-forward on  
  `concat(inSample.tail(train_period), outSample)`.
- Final metrics/plots use an **inner join** of:
  - `highRisk`, `lowRisk`
  - benchmarks `agas`, `close_overal`
  - `equalweight`
- In the shipped outputs, that joined evaluation sample has **180 trading days**  
  (`2024-06-01` → `2025-03-02`).

---

## Repository structure

```text
ETF_portfolio_optimization/
│
├── data/
│   ├── closeData.csv              # ETF close prices (required)
│   └── logRetBenchETF.csv         # Benchmark log returns (required)
│
├── main.py                        # Main end-to-end pipeline
├── gaOpt1.py                      # Inner GA (portfolio weights)
├── gaOpt2.py                      # Outer GA (walk-forward windows)
├── walkforward.py                 # Walk-forward splits and evaluation
├── fitness_ga1.py                 # Fitness / return helpers
├── makeArrayOfWeights.py          # Valid discrete weight set
├── generate_final_results.py      # Metrics, charts, Shapiro / Levene tests
├── equal_weight_train_test.py     # Equal-weight baseline
├── markowitz_train_test.py        # Markowitz (max-Sharpe) baseline module
│
├── analysis_alpha_beta_sensitivity.py   # Optional: alpha/beta sensitivity
├── analysis_nonparametric_tests.py      # Optional: extra statistical tests
├── ga1_parameter_study.py               # Optional: GA1 hyperparameter study
│
├── finalOutputs/                  # Complete results shipped with this repo
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.8+
- Packages in `requirements.txt` (numpy, pandas, scipy, matplotlib, seaborn; optional pyarrow)

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## How to reproduce the main results

### 1. Prepare input data

**`data/closeData.csv`**

- Date column: `date`
- Close columns:
  - `close_afran`, `close_yaghoot` (equity)
  - `close_goldMofid`, `close_nahal`, `close_sahar` (gold)
  - `close_agas`, `close_sarv`, `close_atlas` (fixed income)

**`data/logRetBenchETF.csv`**

- Date column: `date` (format `M/D/YYYY`)
- Required return columns: `agas`, `close_overal`

Log returns for the ETF universe are computed automatically:

```text
log(close_t / close_{t-1})
```

### 2. Run the main pipeline

```bash
python main.py
```

Steps:

1. Load prices and compute log returns.
2. Split data into 80% in-sample / 20% out-of-sample.
3. Enumerate valid portfolio weights.
4. Ask for High Risk / Low Risk coefficients.
5. Run outer GA2 for each profile.
6. Export walk-forward daily returns.
7. Build equal-weight baseline, metrics, charts, and Shapiro/Levene tests.

To reproduce the **shipped High Risk setting**, enter `0.9` / `0.1` (not the 0.8 / 0.2 code default).

**Runtime:** typically several hours. For faster experiments, reduce `population_size` / `generations` in `main.py` and `walkforward.py`.

### 3. Optional robustness / review analyses

Not required for the main pipeline:

```bash
python analysis_alpha_beta_sensitivity.py
python analysis_nonparametric_tests.py
python ga1_parameter_study.py
```

---

## Method overview

```text
GA2 (outer)
  chromosome: [train_period, test_period]
  search space: {10, 20, ..., 200}, with train_period >= test_period
  fitness: mean walk-forward test daily return (in-sample)

    └── for each walk-forward window:
          GA1 (inner)
            chromosome: 8 portfolio weights
            fitness: alpha * return_norm - beta * risk_norm
            apply best weights to the next test window
```

Default GA sizes in code:

| Component | Population | Max generations |
|---|---|---|
| GA2 (outer) | 70 | 40 |
| GA1 (inner, inside walk-forward) | 100 | 50 |

Both GAs use adaptive mutation (`0.05` → `0.10`) and early stopping after stagnation.

### Portfolio constraints

- weights sum to 1
- no short selling (weights ≥ 0)
- equity ≥ 12.5%, gold ≥ 12.5%, fixed income ≥ 12.5%
- discrete weight grid with step `0.1`

---

## Core modules

| File | Purpose |
|---|---|
| `main.py` | Data loading, GA2, return export, final evaluation |
| `gaOpt1.py` | Inner GA for portfolio weights |
| `gaOpt2.py` | Outer GA for walk-forward window lengths |
| `walkforward.py` | Rolling train/test splits; runs GA1 per train window |
| `fitness_ga1.py` | Fitness and daily portfolio returns |
| `makeArrayOfWeights.py` | Feasible weight enumeration |
| `generate_final_results.py` | Merge series; metrics; charts; Shapiro / Levene |
| `equal_weight_train_test.py` | 1/N baseline on the same split |
| `markowitz_train_test.py` | Max-Sharpe baseline module |

---

## Optional analysis scripts

Standalone scripts. The main pipeline does not import them.

### 1) `analysis_alpha_beta_sensitivity.py`

Shows that `(alpha, beta)` behaves as a risk-preference dial.

- Fixed outer window: `train=70`, `test=40`
- Alpha grid: `{0.1, 0.3, 0.5, 0.7, 0.9}` (`beta = 1 - alpha`)
- Inner GA only; two seeds
- Output: `AlphaBeta_Sensitivity.csv`  
  (performance metrics + average equity/gold/fixed-income weights)

### 2) `analysis_nonparametric_tests.py`

Extra distribution-free tests on **already generated** return series (no re-optimization).

Reports:
- Mann–Whitney U
- Kruskal–Wallis
- Paired Wilcoxon signed-rank
- Shapiro–Wilk and Levene re-confirmation
- Recomputed descriptive metrics on the same joined sample

Outputs:
- `Nonparametric_Tests_Results.csv`
- `Table2_metrics_recomputed.csv`

### 3) `ga1_parameter_study.py`

OFAT justification of inner-GA hyperparameters around the published setting  
(`population=100`, `generations=50`, `initial mutation=0.05`).

This study uses `alpha=0.9`, `beta=0.1` and fixed windows `train=50`, `test=20` on **in-sample** data only.

Outputs:
- `GA1_parameter_study.csv`
- `GA1_parameter_study_raw.csv`

---

## Contents of `finalOutputs/`

The repository currently ships **17 files**. Exact inventory:

### A) Main pipeline results

| File | What it contains |
|---|---|
| `optimal_parameters.json` | High/Low Risk coefficients and optimal `(train, test)` windows used for the shipped run |
| `dailyReturn_highRisk.csv` | Walk-forward daily log returns (180 rows; `Date`, `Return`) |
| `dailyReturn_lowRisk.csv` | Walk-forward daily log returns (180 rows; `Date`, `Return`) |
| `equal_weight_test_returns.csv` | Equal-weight baseline returns (196 raw OOS rows before join) |
| `Results_metrics.csv` | Metrics for `highRisk`, `lowRisk`, `agas`, `close_overal`, `equalweight` on the **joined 180-day** sample |
| `Statistical_Tests_Results.csv` | **Shapiro–Wilk** and **Levene** only (main pipeline) |
| `Cum_highRisk.png` | Cumulative return chart (High Risk) |
| `Cum_lowRisk.png` | Cumulative return chart (Low Risk) |
| `Cum_portfolioS.png` | Both GA portfolios |
| `Cum_portfolioS-benchmarkS.png` | GA portfolios + `agas` + `close_overal` + `equalweight` |
| `DD_portfolioS.png` | Drawdowns of both GA portfolios |
| `DD_portfolioS-benchmarkS.png` | Drawdowns including benchmarks |

### B) Optional analysis results

| File | Produced by |
|---|---|
| `AlphaBeta_Sensitivity.csv` | `analysis_alpha_beta_sensitivity.py` |
| `Nonparametric_Tests_Results.csv` | `analysis_nonparametric_tests.py` |
| `Table2_metrics_recomputed.csv` | `analysis_nonparametric_tests.py` |
| `GA1_parameter_study.csv` | `ga1_parameter_study.py` |
| `GA1_parameter_study_raw.csv` | `ga1_parameter_study.py` |

### Consistency notes for these files

1. **Metrics sample:** `Results_metrics.csv` and `Table2_metrics_recomputed.csv` describe the same five series on the joined dates. Values match (rounding differences only).
2. **MDD units differ by file:**
   - `Results_metrics.csv`: MDD as a **fraction** (e.g. `-0.0568`)
   - `Table2_metrics_recomputed.csv` / `AlphaBeta_Sensitivity.csv`: MDD as a **percent** (e.g. `-5.6808`)
3. **Statistical files are complementary, not duplicates:**
   - `Statistical_Tests_Results.csv` → Shapiro + Levene from `generate_final_results.py`
   - `Nonparametric_Tests_Results.csv` → Mann–Whitney, Kruskal–Wallis, Wilcoxon (+ Shapiro/Levene re-check)
4. **Equal-weight raw length (196) > joined length (180):** this is expected; evaluation always uses the inner join.

---

## Notes for reviewers

1. **Stochastic optimization:** GA results can vary across runs unless seeds are fixed. Optional analysis scripts use explicit seeds.
2. **Main vs optional:**
   - Main results: `python main.py`
   - Extra robustness checks: the three optional scripts above
3. **No look-ahead in outer search:** GA2 uses in-sample data only. Reported comparison metrics are out-of-sample (joined dates).
4. **Markowitz:** implemented but currently disabled; not part of shipped `finalOutputs/`.
5. **Computational cost:** nested GA evaluation is expensive because each outer candidate triggers many inner GA runs.

---

**Last updated:** July 2026
