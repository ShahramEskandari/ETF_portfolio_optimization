"""Sharpe ratios under a nonzero risk-free rate (revision round 1).

The revised manuscript reports all Sharpe ratios with a 20% ANNUAL risk-free
rate (reviewer request), converted to a daily log equivalent:

    rf_daily = ln(1 + rf_annual) / days_per_year

where days_per_year is the average number of trading days per year in the
full closeData sample (~238). This script makes every rf-adjusted number in
the paper traceable to an output file:

  1. finalOutputs/Results_metrics_rf20.csv
       The five out-of-sample series (both model portfolios and the three
       benchmarks) with mean, std, Sharpe at rf=0 and rf=20%, cumulative
       return, and max drawdown  ->  source of TABLE 3.
  2. finalOutputs/Sharpe_RF_Sensitivity.csv
       Sharpe ratios of the five series under annual risk-free rates of
       0/15/20/25/30%  ->  source of TABLE 4.
  3. finalOutputs/AlphaBeta_Sensitivity.csv
       A 'sharpe_rf20' column is appended (idempotent, pure arithmetic from
       the existing mean/std columns; no GA re-run). The seed-averaged
       sharpe_rf20 per alpha  ->  Sharpe column of TABLE 5.
       (If analysis_alpha_beta_sensitivity.py is ever re-run, run this
       script again afterwards to restore the column.)

The five return series are reconstructed with the exact inner-join used by
generate_final_results.py (via analysis_nonparametric_tests.load_returns),
so mean/std/cumulative/drawdown match Results_metrics.csv to full precision.
No model code is modified.
"""

import os
import numpy as np
import pandas as pd

from analysis_nonparametric_tests import load_returns

RF_RATES = [0.00, 0.15, 0.20, 0.25, 0.30]
RF_MAIN = 0.20
OUTPUT_DIR = "finalOutputs"
CLOSE_FILE = os.path.join("data", "closeData.csv")
LABELS = {
    "highRisk": "Risk-seeking portfolio",
    "lowRisk": "Risk-averse portfolio",
    "agas": "Agas fund",
    "close_overal": "Overall market index",
    "equalweight": "Equal-weight portfolio",
}


def trading_days_per_year():
    """Average trading days per year over the full closeData sample."""
    close = pd.read_csv(CLOSE_FILE, parse_dates=["date"])
    years = (close["date"].iloc[-1] - close["date"].iloc[0]).days / 365.25
    return (len(close) - 1) / years


def perf_row(r, rf_daily):
    mean, std = r.mean(), r.std()
    cumulative = np.exp(r.cumsum())
    running_max = cumulative.expanding().max()
    return {
        "Mean Daily Log Return": mean,
        "Standard Deviation": std,
        "Sharpe (rf=0)": mean / std,
        f"Sharpe (rf={int(RF_MAIN*100)}%)": (mean - rf_daily) / std,
        "Cumulative Return %": (np.exp(r.sum()) - 1) * 100,
        "Max Drawdown %": ((cumulative - running_max) / running_max).min() * 100,
    }


def main():
    dpy = trading_days_per_year()
    print(f"trading days/year = {dpy:.2f}")
    rf_daily_main = np.log(1 + RF_MAIN) / dpy
    print(f"daily log rf at {int(RF_MAIN*100)}% = {rf_daily_main:.6f}")

    returns = load_returns()
    print(f"merged out-of-sample days = {len(returns)}")

    # ---- 1. Table 3 source ----
    rows = {LABELS[c]: perf_row(returns[c], rf_daily_main) for c in returns.columns}
    df_metrics = pd.DataFrame(rows).T
    path1 = os.path.join(OUTPUT_DIR, "Results_metrics_rf20.csv")
    df_metrics.to_csv(path1, index_label="Series")
    print(f"\n[1] {path1}  (Table 3)")
    print(df_metrics.round(4).to_string())

    # ---- 2. Table 4 source ----
    sens = {}
    for c in returns.columns:
        mean, std = returns[c].mean(), returns[c].std()
        sens[LABELS[c]] = {
            f"rf={int(rf*100)}%": (mean - np.log(1 + rf) / dpy) / std for rf in RF_RATES
        }
    df_sens = pd.DataFrame(sens).T
    path2 = os.path.join(OUTPUT_DIR, "Sharpe_RF_Sensitivity.csv")
    # 6 decimals so that re-rounding to the paper's 3 decimals is unambiguous
    # (4 decimals would double-round, e.g. 0.176470 -> 0.1765 -> 0.177).
    df_sens.round(6).to_csv(path2, index_label="Series")
    print(f"\n[2] {path2}  (Table 4)")
    print(df_sens.round(3).to_string())

    # ---- 3. Table 5 Sharpe column ----
    path3 = os.path.join(OUTPUT_DIR, "AlphaBeta_Sensitivity.csv")
    ab = pd.read_csv(path3)
    ab["sharpe_rf20"] = ((ab["mean_daily_return"] - rf_daily_main)
                         / ab["std_daily_return"]).round(4)
    ab.to_csv(path3, index=False)
    print(f"\n[3] {path3}  ('sharpe_rf20' column appended/updated)")
    print("    seed-averaged sharpe_rf20 per alpha (Table 5 Sharpe column):")
    print(ab.groupby("alpha")["sharpe_rf20"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
