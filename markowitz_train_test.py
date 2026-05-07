"""
Markowitz train/test optimization on close price data.

This module provides a function that:
1) Loads close prices from data/closeData.csv
2) Splits data by a specified date
3) Optimizes long-only Markowitz weights on the train section
4) Tests those fixed weights on the test section
5) Reports test cumulative return and maximum drawdown
"""

import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# GA project asset ordering in closeData.csv
GA_CLOSE_COLUMNS = [
    "close_afran",
    "close_yaghoot",
    "close_goldMofid",
    "close_nahal",
    "close_sahar",
    "close_agas",
    "close_sarv",
    "close_atlas",
]


def _max_drawdown_from_log_returns(log_returns: pd.Series) -> float:
    """Compute max drawdown from a log-return series."""
    cumulative = np.exp(log_returns.cumsum())
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def _neg_sharpe(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free_rate: float) -> float:
    """
    Negative Sharpe ratio objective for minimization.

    risk_free_rate is assumed to be in the same frequency as returns (daily if returns are daily).
    """
    port_ret = float(np.dot(weights, mu))
    port_var = float(np.dot(weights.T, np.dot(cov, weights)))
    port_std = np.sqrt(max(port_var, 1e-16))
    sharpe = (port_ret - risk_free_rate) / port_std
    return -sharpe


def run_markowitz_train_test(
    split_date: str,
    close_data_path: str = "data/closeData.csv",
    date_col: str = "date",
    risk_free_rate: float = 0.0,
    long_only: bool = True,
    enforce_ga_asset_class_constraints: bool = True,
    min_equity_weight: float = 0.0,
    min_gold_weight: float = 0.0,
    min_fixed_income_weight: float = 0.0,
    save_test_returns_csv: bool = False,
    test_returns_output_path: str = "finalOutputs/markowitz_test_returns.csv",
) -> Dict[str, Any]:
    """
    Train Markowitz on data before split_date, test from split_date onward.

    Parameters
    ----------
    split_date : str
        Date string (e.g. '2023-01-01'). Train uses dates < split_date.
        Test uses dates >= split_date.
    close_data_path : str
        Path to close prices CSV.
    date_col : str
        Date column name in the CSV.
    risk_free_rate : float
        Daily risk-free rate for Sharpe optimization (default 0.0).
    long_only : bool
        If True, bounds are [0, 1]. If False, no bounds are used.
    enforce_ga_asset_class_constraints : bool
        If True, enforce the same asset-class minimum constraints used in GA project:
        equity (0:2), gold (2:5), fixed income (5:8).
    min_equity_weight : float
        Minimum total weight for equity class (indices 0-1).
    min_gold_weight : float
        Minimum total weight for gold class (indices 2-4).
    min_fixed_income_weight : float
        Minimum total weight for fixed-income class (indices 5-7).
    save_test_returns_csv : bool
        If True, saves test log returns to CSV.
    test_returns_output_path : str
        Output CSV path for saved test returns (if enabled).

    Returns
    -------
    Dict[str, Any]
        Contains split date, optimized weights, test cumulative return, test max drawdown,
        train/test sample sizes, and optimization diagnostics.
    """
    close_df = pd.read_csv(close_data_path, parse_dates=[date_col])
    close_df = close_df.set_index(date_col).sort_index()

    if close_df.empty:
        raise ValueError("closeData.csv is empty.")

    # Keep only numeric asset columns.
    close_df = close_df.select_dtypes(include=[np.number]).copy()
    if close_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric asset columns for portfolio optimization.")
    # Reorder to GA's exact asset order when all expected columns exist.
    # This guarantees class constraints are applied to the same assets as GA:
    # 0:2 equity, 2:5 gold, 5:8 fixed income.
    if all(col in close_df.columns for col in GA_CLOSE_COLUMNS):
        close_df = close_df[GA_CLOSE_COLUMNS].copy()

    # Compute daily log returns.
    log_returns = np.log(close_df / close_df.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Not enough data to compute returns.")

    split_ts = pd.to_datetime(split_date)
    train_returns = log_returns[log_returns.index < split_ts]
    test_returns = log_returns[log_returns.index >= split_ts]

    if train_returns.shape[0] < 20:
        raise ValueError("Train section is too short. Provide a later split_date (>= 20 train rows).")
    if test_returns.shape[0] < 2:
        raise ValueError("Test section is too short. Provide an earlier split_date.")

    mu = train_returns.mean().values
    cov = train_returns.cov().values
    n_assets = len(mu)
    if enforce_ga_asset_class_constraints and n_assets != 8:
        raise ValueError(
            "GA-aligned asset-class constraints require exactly 8 assets "
            "(indices 0:2 equity, 2:5 gold, 5:8 fixed income)."
        )

    init_w = np.repeat(1.0 / n_assets, n_assets)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if enforce_ga_asset_class_constraints:
        constraints.extend(
            [
                {"type": "ineq", "fun": lambda w: np.sum(w[0:2]) - min_equity_weight},
                {"type": "ineq", "fun": lambda w: np.sum(w[2:5]) - min_gold_weight},
                {"type": "ineq", "fun": lambda w: np.sum(w[5:8]) - min_fixed_income_weight},
            ]
        )
    bounds = [(0.0, 1.0)] * n_assets if long_only else None

    result = minimize(
        _neg_sharpe,
        x0=init_w,
        args=(mu, cov, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    opt_weights = result.x
    # Numerical cleanup.
    opt_weights = np.clip(opt_weights, 0.0, 1.0) if long_only else opt_weights
    opt_weights = opt_weights / np.sum(opt_weights)

    # Test with fixed train-optimized weights.
    test_port_log_ret = test_returns.dot(opt_weights)
    test_cumulative_return = float(np.exp(test_port_log_ret.sum()) - 1.0)
    test_max_drawdown = _max_drawdown_from_log_returns(test_port_log_ret)

    if save_test_returns_csv:
        out_dir = os.path.dirname(test_returns_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({"Return": test_port_log_ret}).to_csv(
            test_returns_output_path, index_label="Date"
        )

    weights_dict = {col: float(w) for col, w in zip(train_returns.columns, opt_weights)}
    class_weight_sums = None
    class_constraints_satisfied = None
    if enforce_ga_asset_class_constraints:
        equity_sum = float(np.sum(opt_weights[0:2]))
        gold_sum = float(np.sum(opt_weights[2:5]))
        fixed_income_sum = float(np.sum(opt_weights[5:8]))
        class_weight_sums = {
            "equity_sum_0_2": equity_sum,
            "gold_sum_2_5": gold_sum,
            "fixed_income_sum_5_8": fixed_income_sum,
        }
        class_constraints_satisfied = {
            "equity_ok": equity_sum >= min_equity_weight - 1e-10,
            "gold_ok": gold_sum >= min_gold_weight - 1e-10,
            "fixed_income_ok": fixed_income_sum >= min_fixed_income_weight - 1e-10,
        }

    return {
        "split_date": str(split_ts.date()),
        "train_start": str(train_returns.index.min().date()),
        "train_end": str(train_returns.index.max().date()),
        "test_start": str(test_returns.index.min().date()),
        "test_end": str(test_returns.index.max().date()),
        "n_train_rows": int(train_returns.shape[0]),
        "n_test_rows": int(test_returns.shape[0]),
        "optimized_weights": weights_dict,
        "test_cumulative_return": test_cumulative_return,
        "test_max_drawdown": test_max_drawdown,
        "constraints": {
            "sum_weights": 1.0,
            "long_only": bool(long_only),
            "ga_asset_class_constraints_enforced": bool(enforce_ga_asset_class_constraints),
            "min_equity_weight": float(min_equity_weight) if enforce_ga_asset_class_constraints else None,
            "min_gold_weight": float(min_gold_weight) if enforce_ga_asset_class_constraints else None,
            "min_fixed_income_weight": float(min_fixed_income_weight) if enforce_ga_asset_class_constraints else None,
        },
        "class_weight_sums": class_weight_sums,
        "class_constraints_satisfied": class_constraints_satisfied,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def markowitz_train_test_project(
    train_returns,
    test_returns,
    risk_free_rate: float = 0.0,
    long_only: bool = True,
    enforce_ga_asset_class_constraints: bool = True,
    min_equity_weight: float = 0.0,
    min_gold_weight: float = 0.0,
    min_fixed_income_weight: float = 0.0,
    save_test_returns_csv: bool = False,
    test_returns_output_path: str = "finalOutputs/markowitz_test_returns.csv",
):

    if train_returns.shape[0] < 20:
        raise ValueError("Train section is too short. Provide a later split_date (>= 20 train rows).")
    if test_returns.shape[0] < 2:
        raise ValueError("Test section is too short. Provide an earlier split_date.")

    mu = train_returns.mean().values
    cov = train_returns.cov().values
    n_assets = len(mu)
    if enforce_ga_asset_class_constraints and n_assets != 8:
        raise ValueError(
            "GA-aligned asset-class constraints require exactly 8 assets "
            "(indices 0:2 equity, 2:5 gold, 5:8 fixed income)."
        )

    init_w = np.repeat(1.0 / n_assets, n_assets)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if enforce_ga_asset_class_constraints:
        constraints.extend(
            [
                {"type": "ineq", "fun": lambda w: np.sum(w[0:2]) - min_equity_weight},
                {"type": "ineq", "fun": lambda w: np.sum(w[2:5]) - min_gold_weight},
                {"type": "ineq", "fun": lambda w: np.sum(w[5:8]) - min_fixed_income_weight},
            ]
        )
    bounds = [(0.0, 1.0)] * n_assets if long_only else None

    result = minimize(
        _neg_sharpe,
        x0=init_w,
        args=(mu, cov, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    opt_weights = result.x
    # Numerical cleanup.
    opt_weights = np.clip(opt_weights, 0.0, 1.0) if long_only else opt_weights
    opt_weights = opt_weights / np.sum(opt_weights)

    # Test with fixed train-optimized weights.
    test_port_log_ret = test_returns.dot(opt_weights)

    if save_test_returns_csv:
        out_dir = os.path.dirname(test_returns_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({"Return": test_port_log_ret}).to_csv(
            test_returns_output_path, index_label="Date"
        )

    return test_port_log_ret

if __name__ == "__main__":
    # Example usage:
    # Train on data before split_date, test from split_date onward.
    summary = run_markowitz_train_test(
        split_date="2024-01-01",
        close_data_path="data/closeData.csv",
        save_test_returns_csv=True,
        test_returns_output_path="finalOutputs/markowitz_test_returns.csv",
    )
    print("Markowitz train/test summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
