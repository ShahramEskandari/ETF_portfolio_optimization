import os
from typing import Dict, Any

import numpy as np
import pandas as pd


def _max_drawdown_from_log_returns(log_returns: pd.Series) -> float:
    cumulative = np.exp(log_returns.cumsum())
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def run_equal_weight_train_test(
    split_date: str,
    close_data_path: str = "data/closeData.csv",
    date_col: str = "date",
    save_test_returns_csv: bool = False,
    test_returns_output_path: str = "finalOutputs/equal_weight_test_returns.csv",
) -> Dict[str, Any]:
    close_df = pd.read_csv(close_data_path, parse_dates=[date_col])
    close_df = close_df.set_index(date_col).sort_index()

    if close_df.empty:
        raise ValueError("closeData.csv is empty.")

    close_df = close_df.select_dtypes(include=[np.number]).copy()
    if close_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric asset columns.")

    log_returns = np.log(close_df / close_df.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Not enough data to compute returns.")

    split_ts = pd.to_datetime(split_date)
    train_returns = log_returns[log_returns.index < split_ts]
    test_returns = log_returns[log_returns.index >= split_ts]

    if train_returns.shape[0] < 1:
        raise ValueError("Train section is empty. Provide a later split_date.")
    if test_returns.shape[0] < 2:
        raise ValueError("Test section is too short. Provide an earlier split_date.")

    n_assets = train_returns.shape[1]
    weights = np.repeat(1.0 / n_assets, n_assets)

    test_port_log_ret = test_returns.dot(weights)
    test_cumulative_return = float(np.exp(test_port_log_ret.sum()) - 1.0)
    test_max_drawdown = _max_drawdown_from_log_returns(test_port_log_ret)

    if save_test_returns_csv:
        out_dir = os.path.dirname(test_returns_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({"Return": test_port_log_ret}).to_csv(
            test_returns_output_path, index_label="Date"
        )

    weights_dict = {col: float(w) for col, w in zip(train_returns.columns, weights)}

    return {
        "split_date": str(split_ts.date()),
        "train_start": str(train_returns.index.min().date()),
        "train_end": str(train_returns.index.max().date()),
        "test_start": str(test_returns.index.min().date()),
        "test_end": str(test_returns.index.max().date()),
        "n_train_rows": int(train_returns.shape[0]),
        "n_test_rows": int(test_returns.shape[0]),
        "equal_weights": weights_dict,
        "test_cumulative_return": test_cumulative_return,
        "test_max_drawdown": test_max_drawdown,
    }

def equal_weight_train_test_project(
    train_returns,
    test_returns,
    save_test_returns_csv: bool = False,
    test_returns_output_path: str = "finalOutputs/equal_weight_test_returns.csv",
):

    if train_returns.shape[0] < 1:
        raise ValueError("Train section is empty. Provide a later split_date.")
    if test_returns.shape[0] < 2:
        raise ValueError("Test section is too short. Provide an earlier split_date.")

    n_assets = train_returns.shape[1]
    weights = np.repeat(1.0 / n_assets, n_assets)

    test_port_log_ret = test_returns.dot(weights)

    if save_test_returns_csv:
        out_dir = os.path.dirname(test_returns_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({"Return": test_port_log_ret}).to_csv(
            test_returns_output_path, index_label="Date"
        )

    return test_port_log_ret


if __name__ == "__main__":
    summary = run_equal_weight_train_test(
        split_date="2024-01-01",
        close_data_path="data/closeData.csv",
        save_test_returns_csv=True,
        test_returns_output_path="finalOutputs/equal_weight_test_returns.csv",
    )
    print(summary["test_cumulative_return"])
    print(summary["test_max_drawdown"])
