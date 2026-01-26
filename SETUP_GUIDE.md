# ETFs Portfolio Optimization - Setup Guide

## Prerequisites
- Python 3.8 or higher installed on your system
- pip (Python package installer)

## Setting Up Virtual Environment

### Step 1: Create Virtual Environment

Open PowerShell or Command Prompt in your project directory and run:

```powershell
# Using venv (built-in Python module)
python -m venv venv
```

### Step 2: Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
.\venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**Note:** If you get an execution policy error on Windows PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Upgrade pip (Recommended)

```powershell
python -m pip install --upgrade pip
```

### Step 4: Install Required Packages

```powershell
pip install -r requirements.txt
```

### Step 5: Verify Installation

Check that all packages are installed correctly:

```powershell
pip list
```

You should see:
- numpy
- pandas
- matplotlib
- seaborn
- pyarrow

### Step 6: Test Your Setup

Run a simple test to ensure everything works:

```powershell
python -c "import numpy; import pandas; import matplotlib; import seaborn; print('All packages installed successfully!')"
```

## Deactivating Virtual Environment

When you're done working, deactivate the virtual environment:

```powershell
deactivate
```

## Running Your Project

1. **Activate the virtual environment** (if not already active)
2. **Run your main script:**
   ```powershell
   python main.py
   ```

## Project Structure

```
ETFs_portfolio_optimization/
├── venv/                      # Virtual environment (don't commit to git)
├── data/                      # Data files
│   ├── data.csv
│   ├── logRetData.csv
│   ├── simpleRetData.csv
│   └── prepare_data.ipynb
├── main.py                    # Main execution script
├── gaOpt1.py                  # Genetic Algorithm Optimizer 1
├── gaOpt2.py                  # Genetic Algorithm Optimizer 2
├── fitness_ga1.py             # Fitness functions
├── makeArrayOfWeights.py      # Weight generation utilities
├── walkforward.py             # Walk-forward optimization
├── checkOutlier.py            # Outlier detection and visualization
├── requirements.txt           # Python dependencies
└── SETUP_GUIDE.md            # This file

```

## Troubleshooting

### Issue: "pip is not recognized"
**Solution:** Make sure Python is added to your PATH, or use `python -m pip` instead of `pip`

### Issue: "Execution policy error" on Windows
**Solution:** Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Package installation fails
**Solution:** 
1. Make sure you have the latest pip: `python -m pip install --upgrade pip`
2. Try installing packages one by one to identify the problematic package
3. Check your internet connection

### Issue: Import errors when running scripts
**Solution:** 
1. Make sure the virtual environment is activated
2. Verify packages are installed: `pip list`
3. Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

## Additional Notes

- Always activate the virtual environment before working on the project
- Don't commit the `venv/` folder to version control (add it to `.gitignore`)
- If you add new packages, update `requirements.txt` using:
  ```powershell
  pip freeze > requirements.txt
  ```

## Quick Reference Commands

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt

# Run main script
python main.py

# Deactivate venv
deactivate
```

