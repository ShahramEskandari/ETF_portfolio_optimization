"""
Standalone statistical analysis of the out-of-sample daily return series
that produced the paper's Table 2.

This script DOES NOT touch or re-run any model code. It only re-loads the
exact same series files that `generate_final_results.py` merges, reconstructs
the identical inner-joined out-of-sample DataFrame, and then runs:

  Task 1 (non-parametric / distributional tests):
    a) Mann-Whitney U (two-sided) for each portfolio vs each benchmark  (6 tests)
    b) Kruskal-Wallis H across all five series
    c) Paired Wilcoxon signed-rank on daily return differences          (6 tests)
    d) Re-confirmation of Shapiro-Wilk and Levene p-values

  Task 2 (Table 2 descriptive figures):
    mean, std, Sharpe (rf=0), cumulative return %, max drawdown %,
    number of out-of-sample trading days, and exact date range.

Outputs are printed as clean plain-text tables and also written to CSV files
in finalOutputs/ (new files only; existing model outputs are not modified).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, wilcoxon, shapiro, levene

OUTPUT_DIR = "finalOutputs"
BENCHMARK_FILE = "data/logRetBenchETF.csv"

PORTFOLIOS = ["highRisk", "lowRisk"]
BENCHMARKS = ["agas", "close_overal", "equalweight"]
# Human-readable benchmark labels for reporting.
LABELS = {
    "highRisk": "highRisk",
    "lowRisk": "lowRisk",
    "agas": "AGAS fund",
    "close_overal": "total market index (close_overal)",
    "equalweight": "equal-weight",
}


def load_returns():
    """Reconstruct the EXACT merged out-of-sample returns DataFrame used by
    generate_final_results.py (inner join of the five series, then dropna)."""
    # Benchmarks: agas + total market index
    df_index = pd.read_csv(BENCHMARK_FILE)
    df_index["date"] = pd.to_datetime(df_index["date"], format="%m/%d/%Y")
    df_index = df_index.set_index("date")
    df_index = df_index[["agas", "close_overal"]]
    df_index.index.name = "Date"

    portfolio_files = {
        "highRisk": os.path.join(OUTPUT_DIR, "dailyReturn_highRisk.csv"),
        "lowRisk": os.path.join(OUTPUT_DIR, "dailyReturn_lowRisk.csv"),
        "equalweight": os.path.join(OUTPUT_DIR, "equal_weight_test_returns.csv"),
    }

    dfs = {}
    for key, filepath in portfolio_files.items():
        df = pd.read_csv(filepath, parse_dates=["Date"], index_col=["Date"])
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        df = df.rename(columns={"Return": key})
        dfs[key] = df

    # Same join order as generate_final_results.py, how='inner', then dropna.
    df_merged = dfs["highRisk"].join(
        [dfs["lowRisk"], df_index, dfs["equalweight"]], how="inner"
    )
    returns = df_merged.copy().dropna()
    # Column order matching the five series in Table 2.
    returns = returns[["highRisk", "lowRisk", "agas", "close_overal", "equalweight"]]
    return returns


def max_drawdown(returns_series):
    """Max drawdown (fraction, negative) from a daily LOG-return series,
    matching generate_final_results.calculate_metrics."""
    cumulative = np.exp(returns_series.cumsum())
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def table2_metrics(returns):
    """Recompute Table 2 descriptive metrics (4 dp)."""
    rows = []
    for col in returns.columns:
        r = returns[col]
        mean = r.mean()
        std = r.std()  # pandas default ddof=1, matching model code
        sharpe = mean / std  # rf = 0
        cum_ret_pct = (np.exp(r.sum()) - 1) * 100
        mdd_pct = max_drawdown(r) * 100
        rows.append(
            {
                "Series": LABELS[col],
                "Mean Daily Return": round(mean, 4),
                "Std Daily Return": round(std, 4),
                "Sharpe (rf=0)": round(sharpe, 4),
                "Cumulative Return %": round(cum_ret_pct, 4),
                "Max Drawdown %": round(mdd_pct, 4),
            }
        )
    return pd.DataFrame(rows)


def run_mann_whitney(returns):
    rows = []
    for p in PORTFOLIOS:
        for b in BENCHMARKS:
            u, pval = mannwhitneyu(returns[p], returns[b], alternative="two-sided")
            rows.append(
                {
                    "Comparison": f"{LABELS[p]} vs {LABELS[b]}",
                    "U statistic": round(u, 4),
                    "p-value": pval,
                    "Significant (a=0.05)": "Yes" if pval < 0.05 else "No",
                }
            )
    return pd.DataFrame(rows)


def run_kruskal(returns):
    h, pval = kruskal(*[returns[c] for c in returns.columns])
    return pd.DataFrame(
        [
            {
                "Test": "Kruskal-Wallis (all 5 series)",
                "H statistic": round(h, 4),
                "p-value": pval,
                "Significant (a=0.05)": "Yes" if pval < 0.05 else "No",
            }
        ]
    )


def run_wilcoxon(returns):
    """Paired Wilcoxon signed-rank on the daily return DIFFERENCES.
    Series are already aligned by date (shared index of the inner join)."""
    rows = []
    for p in PORTFOLIOS:
        for b in BENCHMARKS:
            diff = (returns[p] - returns[b]).values
            stat, pval = wilcoxon(diff)  # two-sided by default
            rows.append(
                {
                    "Comparison": f"{LABELS[p]} - {LABELS[b]}",
                    "W statistic": round(stat, 4),
                    "p-value": pval,
                    "Significant (a=0.05)": "Yes" if pval < 0.05 else "No",
                }
            )
    return pd.DataFrame(rows)


def run_shapiro(returns):
    rows = []
    for col in returns.columns:
        stat, pval = shapiro(returns[col])
        rows.append(
            {
                "Series": LABELS[col],
                "W statistic": round(stat, 6),
                "p-value": pval,
                "Interpretation": "Normal" if pval > 0.05 else "Not Normal",
            }
        )
    return pd.DataFrame(rows)


def run_levene(returns):
    # Same pairs cited in generate_final_results.perform_statistical_tests.
    test_pairs = [
        ("highRisk", "lowRisk"),
        ("agas", "highRisk"),
        ("agas", "lowRisk"),
        ("highRisk", "equalweight"),
        ("lowRisk", "equalweight"),
    ]
    rows = []
    for a, b in test_pairs:
        stat, pval = levene(returns[a], returns[b])
        rows.append(
            {
                "Pair": f"{a} vs {b}",
                "Statistic": round(stat, 6),
                "p-value": pval,
                "Interpretation": "Equal Variances" if pval > 0.05 else "Different Variances",
            }
        )
    return pd.DataFrame(rows)


def fmt(df):
    """Pretty-print a DataFrame with p-values in scientific notation."""
    d = df.copy()
    for c in d.columns:
        if "p-value" in c:
            d[c] = d[c].map(lambda x: f"{x:.4e}")
    return d.to_string(index=False)


def main():
    returns = load_returns()

    n_days = len(returns)
    date_start = returns.index.min().date()
    date_end = returns.index.max().date()

    print("=" * 90)
    print("OUT-OF-SAMPLE STATISTICAL ANALYSIS  (reconstruction of Table 2 series)")
    print("=" * 90)
    print(f"Out-of-sample trading days : {n_days}")
    print(f"Date range                 : {date_start}  ->  {date_end}")
    print(f"Series                     : {', '.join(returns.columns)}")

    # ---- Task 2: Table 2 descriptive metrics ----
    print("\n" + "=" * 90)
    print("TASK 2 — TABLE 2 DESCRIPTIVE METRICS (4 decimal places)")
    print("=" * 90)
    t2 = table2_metrics(returns)
    print(t2.to_string(index=False))

    # ---- Task 1a: Mann-Whitney U ----
    print("\n" + "=" * 90)
    print("TASK 1a — MANN-WHITNEY U TESTS (two-sided): portfolios vs benchmarks")
    print("=" * 90)
    mw = run_mann_whitney(returns)
    print(fmt(mw))

    # ---- Task 1b: Kruskal-Wallis ----
    print("\n" + "=" * 90)
    print("TASK 1b — KRUSKAL-WALLIS H TEST (across all five series)")
    print("=" * 90)
    kw = run_kruskal(returns)
    print(fmt(kw))

    # ---- Task 1c: Paired Wilcoxon signed-rank ----
    print("\n" + "=" * 90)
    print("TASK 1c — PAIRED WILCOXON SIGNED-RANK TESTS (on daily return differences)")
    print("=" * 90)
    wx = run_wilcoxon(returns)
    print(fmt(wx))

    # ---- Task 1d: Shapiro-Wilk & Levene re-confirmation ----
    print("\n" + "=" * 90)
    print("TASK 1d — SHAPIRO-WILK (normality) — re-confirmation")
    print("=" * 90)
    sw = run_shapiro(returns)
    print(fmt(sw))

    print("\n" + "=" * 90)
    print("TASK 1d — LEVENE (variance equality) — re-confirmation")
    print("=" * 90)
    lv = run_levene(returns)
    print(fmt(lv))

    # ---- Persist to CSV (new files only) ----
    summary_path = os.path.join(OUTPUT_DIR, "Table2_metrics_recomputed.csv")
    tests_path = os.path.join(OUTPUT_DIR, "Nonparametric_Tests_Results.csv")
    t2.to_csv(summary_path, index=False)

    # Combine all test tables into one tidy CSV.
    combined = []
    for _, row in mw.iterrows():
        combined.append({"Test": "Mann-Whitney U", "Detail": row["Comparison"],
                         "Statistic": row["U statistic"], "p-value": row["p-value"],
                         "Significant": row["Significant (a=0.05)"]})
    for _, row in kw.iterrows():
        combined.append({"Test": "Kruskal-Wallis", "Detail": "all 5 series",
                         "Statistic": row["H statistic"], "p-value": row["p-value"],
                         "Significant": row["Significant (a=0.05)"]})
    for _, row in wx.iterrows():
        combined.append({"Test": "Wilcoxon signed-rank", "Detail": row["Comparison"],
                         "Statistic": row["W statistic"], "p-value": row["p-value"],
                         "Significant": row["Significant (a=0.05)"]})
    for _, row in sw.iterrows():
        combined.append({"Test": "Shapiro-Wilk", "Detail": row["Series"],
                         "Statistic": row["W statistic"], "p-value": row["p-value"],
                         "Significant": ""})
    for _, row in lv.iterrows():
        combined.append({"Test": "Levene", "Detail": row["Pair"],
                         "Statistic": row["Statistic"], "p-value": row["p-value"],
                         "Significant": ""})
    pd.DataFrame(combined).to_csv(tests_path, index=False)

    print("\n" + "=" * 90)
    print("CSV outputs written:")
    print(f"  - {summary_path}")
    print(f"  - {tests_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
