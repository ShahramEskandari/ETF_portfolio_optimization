# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Setup Environment (5 minutes)

```bash
# Clone repository
git clone <your-repo-url>
cd ETFs_portfolio_optimization

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source venv/bin/activate      # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

### Step 2: Prepare Data (2 minutes)

1. Ensure `data/data.csv` exists (raw ETF prices)
2. Open `data/prepare_data.ipynb` in Jupyter
3. Run all cells → generates `logRetData.csv`

---

### Step 3: Run Optimization

#### Option A: Find Optimal Parameters (SLOW - several hours)
```bash
python main.py
```

#### Option B: Use Pre-determined Parameters (FAST - 1-2 hours)
```bash
python make_csv_final_results.py
```

Then move results and visualize:
```bash
move dailyReturn_*.csv "plot results/"
# Open plot results/evaluationOfResults.ipynb and run all cells
```

---

## 📊 What You'll Get

- **3 Portfolio Strategies**: High-risk, Low-risk, Normal-risk
- **Performance Metrics**: Returns, Sharpe ratio, drawdown, etc.
- **Visualizations**: Cumulative returns, drawdown charts
- **Statistical Analysis**: Hypothesis tests, comparisons

---

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `main.py` | Find optimal walk-forward parameters |
| `make_csv_final_results.py` | Generate portfolio returns |
| `data/prepare_data.ipynb` | Prepare return data |
| `plot results/evaluationOfResults.ipynb` | Analyze & visualize |

---

## ⚙️ Default Parameters

```python
High Risk:   train=50,  test=5   # Aggressive
Low Risk:    train=250, test=10  # Conservative  
Normal Risk: train=245, test=35  # Balanced
```

---

## 📖 Need More Details?

See [README.md](README.md) for complete documentation.

---

## ❓ Common Issues

**Issue**: "Module not found"
**Fix**: Make sure venv is activated and dependencies installed

**Issue**: "File not found: logRetData.csv"
**Fix**: Run `data/prepare_data.ipynb` first

**Issue**: Code runs forever
**Fix**: This is normal! GA optimization takes hours. Consider using `make_csv_final_results.py` instead.

---

**Happy Optimizing! 🎉**

