# ETF Portfolio Optimization using Nested Genetic Algorithms

A sophisticated portfolio optimization system that uses **nested genetic algorithms** to find optimal ETF portfolio weights and walk-forward analysis parameters. The system optimizes both the portfolio composition and the time-window parameters for walk-forward validation.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage Workflow](#usage-workflow)
- [File Descriptions](#file-descriptions)
- [Algorithm Details](#algorithm-details)
- [Results & Visualization](#results--visualization)
- [Portfolio Constraints](#portfolio-constraints)

---

## 🎯 Overview

This project implements a **two-level genetic algorithm optimization**:

1. **Outer GA (GA2)**: Optimizes walk-forward analysis parameters (train period, test period)
2. **Inner GA (GA1)**: Optimizes portfolio weights for 8 ETFs

The system evaluates different portfolio strategies (high-risk, low-risk, normal-risk) and compares them against benchmark indices.

### Key Features
- ✅ Nested genetic algorithm optimization
- ✅ Walk-forward analysis for robust backtesting
- ✅ Portfolio constraint enforcement
- ✅ Risk-adjusted return optimization
- ✅ Comprehensive performance metrics
- ✅ Statistical analysis and visualization

---

## 🏗️ Project Architecture

```
ETFs_portfolio_optimization/
│
├── data/                              # Data directory
│   ├── data.csv                       # Raw ETF price data (INPUT)
│   ├── prepare_data.ipynb             # Data preprocessing notebook
│   ├── logRetData.csv                 # Log returns (GENERATED - used in optimization)
│   └── simpleRetData.csv              # Simple returns (GENERATED)
│
├── Core Algorithm Files
│   ├── main.py                        # Main execution script (GA2 optimization)
│   ├── gaOpt1.py                      # Inner GA: Portfolio weight optimization
│   ├── gaOpt2.py                      # Outer GA: Walk-forward parameter optimization
│   ├── fitness_ga1.py                 # Fitness functions for portfolio evaluation
│   ├── walkforward.py                 # Walk-forward analysis implementation
│   └── makeArrayOfWeights.py          # Generate valid portfolio weight combinations
│
├── Results Generation
│   └── make_csv_final_results.py      # Generate daily returns for final portfolios
│
├── plot results/                      # Results and visualization directory
│   ├── dailyReturn_highRisk.csv       # High-risk portfolio daily returns
│   ├── dailyReturn_lowRisk.csv        # Low-risk portfolio daily returns
│   ├── dailyReturn_normalRisk.csv     # Normal-risk portfolio daily returns
│   ├── evaluationOfResults.ipynb      # Statistical analysis & visualization
│   ├── logRetBenchETF.csv             # Benchmark ETF returns
│   ├── Results_metrics.csv            # Performance metrics summary
│   ├── Cum_*.png                      # Cumulative return charts
│   └── DD_*.png                       # Drawdown charts
│
├── Utility Files
│   ├── checkOutlier.py                # Outlier detection visualization
│   ├── requirements.txt               # Python dependencies
│   ├── SETUP_GUIDE.md                 # Detailed setup instructions
│   └── README.md                      # This file
│
└── venv/                              # Virtual environment (not in git)
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ETFs_portfolio_optimization
```

2. **Create virtual environment**
```bash
python -m venv venv
```

3. **Activate virtual environment**

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Required Packages:**
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `scipy` - Statistical tests (for evaluation)
- `pyarrow` - Better pandas performance

---

## 📊 Usage Workflow

The project follows a **3-step workflow**:

### **STEP 1: Data Preparation**

**Input:** `data/data.csv` (raw ETF closing prices)

**Process:**
1. Open `data/prepare_data.ipynb`
2. Run all cells to:
   - Load raw price data
   - Calculate log returns
   - Calculate simple returns
   - Rename columns appropriately

**Output:**
- `data/logRetData.csv` - Log returns (used in optimization)
- `data/simpleRetData.csv` - Simple returns

**Data Structure:**
- **Columns**: 8 ETF return series
  - `ret_afran`, `ret_yaghoot`, `ret_goldMofid`, `ret_nahal`
  - `ret_sahar`, `ret_agas`, `ret_sarv`, `ret_atlas`
- **Index**: Date (daily frequency)

---

### **STEP 2: Optimization (Main GA Flow)**

**Input:** `data/logRetData.csv`

**Process:**
Run the main optimization to find optimal walk-forward parameters:

```bash
python main.py
```

**What happens:**
1. **Outer GA (GA2)** searches for optimal:
   - `train_period`: Training window size (5-250 days, steps of 5)
   - `test_period`: Testing window size (5-250 days, steps of 5)
   - Constraint: `train_period >= test_period`

2. **Inner GA (GA1)** (called by walk-forward analysis):
   - For each train/test split:
     - Optimizes 8 ETF portfolio weights
     - Enforces portfolio constraints
     - Evaluates on test period

3. **Fitness Evaluation:**
   - Walk-forward validation across entire dataset
   - Calculates mean daily return across all test periods

**Output:**
- Console output showing:
  - Best parameters per generation
  - Final optimal `[train_period, test_period]`
  - Best fitness value (mean return)

**Example Output:**
```
*** GA2 *** : generation 0 : parameters = [245  35] , cost = 0.001234
*** GA2 *** : generation 1 : parameters = [250  10] , cost = 0.001456
...
best param = [245  35]
best cost = 0.001567
```

---

### **STEP 3: Generate Results & Visualization**

#### **3A. Generate Daily Returns for Final Portfolios**

After finding optimal parameters, generate daily returns for three risk profiles:

```bash
python make_csv_final_results.py
```

**This script:**
- Uses pre-determined optimal parameters for:
  - **High Risk**: `train=50, test=5` (short windows, frequent rebalancing)
  - **Low Risk**: `train=250, test=10` (long windows, stable weights)
  - **Normal Risk**: `train=245, test=35` (balanced approach)

- Generates CSV files with daily returns:
  - `dailyReturn_highRisk.csv`
  - `dailyReturn_lowRisk.csv`
  - `dailyReturn_normalRisk.csv`

**Move files to plot results folder:**
```bash
# Windows
move dailyReturn_*.csv "plot results/"

# Linux/macOS
mv dailyReturn_*.csv "plot results/"
```

---

#### **3B. Analyze Results & Generate Visualizations**

**Input Files in `plot results/`:**
- `dailyReturn_highRisk.csv`
- `dailyReturn_lowRisk.csv`
- `dailyReturn_normalRisk.csv`
- `logRetBenchETF.csv` (benchmark data)

**Process:**
1. Navigate to `plot results/` folder
2. Open `evaluationOfResults.ipynb`
3. Run all cells

**What it does:**
- Calculates performance metrics:
  - Cumulative returns
  - Sharpe ratio
  - Maximum drawdown
  - Volatility
  - Win rate
- Statistical tests:
  - Normality tests
  - Variance equality tests
  - Mann-Whitney U tests
- Generates visualizations:
  - Cumulative return charts
  - Drawdown charts
  - Distribution plots
  - Comparison charts

**Output Files:**
- `Cum_highRisk.png` - Cumulative returns (high risk)
- `Cum_lowRisk.png` - Cumulative returns (low risk)
- `Cum_normalRisk.png` - Cumulative returns (normal risk)
- `Cum_portfolioS.png` - All portfolios comparison
- `DD_portfolioS.png` - Drawdown comparison
- `DD_normalRisk.png` - Normal risk drawdown detail
- `Results_metrics.csv` - Summary statistics table

---

## 📁 File Descriptions

### Core Algorithm Files

#### `main.py`
Main execution script that runs GA2 optimization.
- Loads log return data
- Generates valid weight combinations
- Runs outer GA to find optimal walk-forward parameters
- Prints best solution

#### `gaOpt1.py` - Inner Genetic Algorithm
Optimizes portfolio weights (8 ETFs).

**Key Functions:**
- `generate_gene()`: Creates random valid portfolio weights
- `create_population()`: Initializes population
- `selection()`: Tournament selection based on fitness
- `crossover()`: Single-point crossover with normalization
- `mutation()`: Random gene mutation with constraint checking
- `GA()`: Main GA loop with adaptive mutation rate

**Features:**
- Adaptive mutation rate (0.3 → 0.8 when stuck)
- Early stopping when converged
- Constraint enforcement at every step

#### `gaOpt2.py` - Outer Genetic Algorithm
Optimizes walk-forward parameters (train/test periods).

**Key Functions:**
- `generate_gene()`: Creates random [train_period, test_period]
- `selection()`: Calls walk-forward analysis for fitness
- `crossover()`: Integer crossover for period parameters
- `mutation()`: Mutates periods within valid ranges
- `GA2()`: Main GA loop

**Chromosome Structure:**
```python
[train_period, test_period]  # Both in range [5, 250], steps of 5
```

#### `walkforward.py`
Implements walk-forward analysis.

**Key Functions:**
- `walk_forward_split()`: Splits data into train/test windows
- `walkForwardOptimization3()`: 
  - Main walk-forward function
  - Runs GA1 on each train period
  - Evaluates on corresponding test period
  - Optionally saves daily returns to CSV
  - Returns mean return across all test periods

**Walk-Forward Process:**
```
Data: [-----------------------------]
Split 1: [Train 1][Test 1]
Split 2:      [Train 2][Test 2]
Split 3:           [Train 3][Test 3]
...
```

#### `fitness_ga1.py`
Portfolio evaluation functions.

**Functions:**
- `fitness_function()`: Scaled fitness (return - risk)
  - Normalizes return and risk to [0,1]
  - Weighted combination: `0.8*return - 0.2*risk`
  
- `calc_riskAdjRet()`: Sharpe-like ratio (mean/std)

- `calc_returnOfPoints()`: Daily portfolio returns
  - Returns array of daily returns for test period
  - Used for walk-forward evaluation

- `findMinMax()`: Finds extrema for normalization
  - Calculates min/max return and risk across all valid weights

#### `makeArrayOfWeights.py`
Generates all valid portfolio weight combinations.

**Function:**
- `make_weights()`: 
  - Generates all combinations with 0.1 step size
  - Filters by constraints
  - Returns ~165,000 valid combinations

- `is_valid()`: Checks portfolio constraints

---

### Utility Files

#### `make_csv_final_results.py`
Generates daily returns for three risk profiles using pre-determined optimal parameters.

**Parameters Used:**
```python
High Risk:   train=50,  test=5   # Aggressive, frequent rebalancing
Low Risk:    train=250, test=10  # Conservative, stable weights
Normal Risk: train=245, test=35  # Balanced approach
```

#### `checkOutlier.py`
Visualizes outliers in return data using box plots.

```bash
python checkOutlier.py
```

Generates: `outliers of return.png`

---

## 🧬 Algorithm Details

### Nested GA Architecture

```
GA2 (Outer)
├─ Chromosome: [train_period, test_period]
├─ Fitness: Mean return from walk-forward analysis
└─ Evaluation:
    └─ Walk-Forward Analysis
        ├─ Split data into train/test windows
        └─ For each window:
            ├─ GA1 (Inner)
            │   ├─ Chromosome: [w1, w2, ..., w8] (8 ETF weights)
            │   ├─ Fitness: Scaled return-risk
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

3. **Non-negativity**: `all weights ≥ 0`

**ETF Categories:**
```python
Equity:       [afran, yaghoot]           # Indices 0-1
Gold:         [goldMofid, nahal, sahar]  # Indices 2-4
Fixed Income: [agas, sarv, atlas]        # Indices 5-7
```

### Fitness Function

**Inner GA (Portfolio Weights):**
```python
fitness = 0.8 * return_scaled - 0.2 * risk_scaled

where:
  return_scaled = (mean_return - min_return) / (max_return - min_return)
  risk_scaled = (portfolio_std - min_std) / (max_std - min_std)
```

**Outer GA (Walk-Forward Parameters):**
```python
fitness = mean(daily_returns_across_all_test_periods)
```

### Adaptive Mutation

Both GAs use adaptive mutation rates:
- **Initial**: 0.3 (30% mutation probability)
- **High**: 0.8 (80% mutation probability)

**Trigger Condition:**
If best fitness doesn't improve for 8 consecutive generations:
- Switch to high mutation rate (exploration)
- If still no improvement after high mutation: stop (converged)

---

## 📈 Results & Visualization

### Performance Metrics

The `evaluationOfResults.ipynb` calculates:

1. **Return Metrics**:
   - Total return
   - Annualized return
   - Mean daily return

2. **Risk Metrics**:
   - Volatility (daily, annualized)
   - Maximum drawdown
   - Downside deviation

3. **Risk-Adjusted Metrics**:
   - Sharpe ratio
   - Sortino ratio
   - Calmar ratio

4. **Other Metrics**:
   - Win rate
   - Best/worst day
   - Recovery time

### Statistical Tests

1. **Normality Tests**:
   - Shapiro-Wilk test
   - Jarque-Bera test

2. **Variance Tests**:
   - Levene's test
   - Bartlett's test

3. **Mean Comparison**:
   - Mann-Whitney U test (non-parametric)
   - t-test (parametric)

### Visualization Outputs

**Cumulative Return Charts:**
- Shows portfolio growth over time
- Compares against benchmarks
- Identifies periods of outperformance/underperformance

**Drawdown Charts:**
- Visualizes peak-to-trough declines
- Shows recovery periods
- Compares drawdown severity across portfolios

**Distribution Plots:**
- Return distributions (histograms, KDE)
- Q-Q plots for normality
- Box plots for outliers

---

## 🔧 Configuration

### Modifying GA Parameters

**In `gaOpt1.py` and `gaOpt2.py`:**
```python
# Population size and generations
GA(population_size=70, generations=40, ...)  # Inner GA
GA2(population_size=30, generations=10, ...) # Outer GA

# Mutation rates
initial_mutation_rate = 0.3
high_mutation_rate = 0.8
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
weightReturn = 0.8  # Return importance
weightRisk = 0.2    # Risk importance
```

---

## 🎓 Understanding the Results

### Interpreting GA2 Output

```
*** GA2 *** : generation 5 : parameters = [245  35] , cost = 0.001567
```
- **parameters**: `[train_period=245 days, test_period=35 days]`
- **cost**: Mean daily return = 0.1567% per day

### Risk Profile Comparison

**High Risk (50/5):**
- Short training window → responsive to recent data
- Short test window → frequent rebalancing
- Higher returns but higher volatility

**Low Risk (250/10):**
- Long training window → stable, smooth weights
- Moderate test window → less frequent rebalancing
- Lower returns but lower volatility

**Normal Risk (245/35):**
- Long training window → stable optimization
- Longer test window → balanced rebalancing frequency
- Moderate risk-return tradeoff

---

## 📝 Notes & Best Practices

1. **Computation Time**:
   - `main.py` can take several hours (nested GAs)
   - `make_csv_final_results.py` takes 1-2 hours per portfolio
   - Consider running overnight or on a powerful machine

2. **Random Seed**:
   - Results may vary between runs (stochastic optimization)
   - Set `np.random.seed()` for reproducibility

3. **Data Requirements**:
   - Minimum 500 days of data recommended
   - Daily frequency required
   - No missing values in return series

4. **Memory Usage**:
   - `make_weights()` generates ~165K combinations (~10MB)
   - Consider reducing step size for memory constraints

5. **Overfitting Prevention**:
   - Walk-forward analysis provides out-of-sample validation
   - Multiple test periods reduce overfitting risk
   - Compare against simple benchmarks (equal weight, buy-hold)

---

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

---

## 📄 License

[Add your license here]

---

## 👤 Author

[Add your name and contact]

---

## 🙏 Acknowledgments

This project implements concepts from:
- Genetic algorithms for portfolio optimization
- Walk-forward analysis for robust backtesting
- Modern portfolio theory
- Risk-adjusted performance metrics

---

## 📚 References

1. Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*.
2. Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*.
3. Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*.

---

**Last Updated**: 2025

**Version**: 1.0

---

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)

