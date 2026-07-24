"""
alpha/beta sensitivity analysis for the inner-GA fitness function

    fitness = alpha * return_norm - beta * risk_norm ,   beta = 1 - alpha

The reviewer questioned the separate min-max normalization of the return and
risk terms. This standalone script demonstrates that (alpha, beta) still acts
as an interpretable risk-preference dial: as alpha increases, realized
out-of-sample risk (std, max drawdown) rises and the portfolio tilts away from
fixed income toward equity.

DESIGN (fallback option b, chosen because the full nested pipeline would take
~10-15h for 5 alpha x 2 seeds):
  * The OUTER-layer walk-forward window is FIXED to a single neutral
    configuration (train=70, test=40 trading days) for EVERY alpha. Holding the
    window constant isolates the effect of alpha.
  * Only the INNER GA runs, via the model's own GA()/walk_forward_split()/
    calc_returnOfPoints() functions. This script REPLICATES the exact loop in
    walkforward.walkForwardOptimization3 (same GA(100,50), same best-solution
    selection, same return computation) and additionally records the winning
    weight vector per window so we can report asset-class allocations.
  * No model code is modified; this script only calls the existing functions.

Out-of-sample definition matches the paper: time_series_split(test_size=0.2)
on the daily log returns; the walk-forward slides over
pd.concat([inSample.tail(70), outSample]) exactly as main.py does. With
train=70 the test windows begin precisely at the first out-of-sample day, so
all reported test days are genuinely out-of-sample (first 160 of 196 OOS days;
fewer than the paper's 180 because the window is fixed rather than GA-optimized
and there is no benchmark inner-join).

Seeds: each (alpha) is run twice, with SEED_A and SEED_B below, seeding both
numpy and the random module (the GA uses both). Reported per seed so result
stability can be judged.
"""

import os
import io
import contextlib
import random
import numpy as np
import pandas as pd

from gaOpt1 import GA
from makeArrayOfWeights import make_weights
from walkforward import walk_forward_split
from fitness_ga1 import calc_returnOfPoints

# ----------------------------- configuration -----------------------------
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]     # alpha = weight_return ; beta = 1 - alpha
SEED_A = 42
SEED_B = 7
TRAIN_PERIOD = 70                       # fixed neutral outer window
TEST_PERIOD = 40
TEST_SIZE = 0.20
OUTPUT_DIR = "finalOutputs"
OUT_CSV = os.path.join(OUTPUT_DIR, "AlphaBeta_Sensitivity.csv")

# Asset-class index map (column order of closeData.csv / log returns):
#   equity        = afran(0), yaghoot(1)
#   gold          = goldMofid(2), nahal(3), sahar(4)
#   fixed income  = agas(5), sarv(6), atlas(7)
EQUITY_IDX = [0, 1]
GOLD_IDX = [2, 3, 4]
FIXED_INCOME_IDX = [5, 6, 7]


def load_split():
    """Reproduce main.py's data loading and 80/20 time split exactly."""
    close_data = pd.read_csv("data/closeData.csv", index_col=["date"], parse_dates=["date"])
    log_returns = np.log(close_data / close_data.shift(1)).dropna()
    log_returns.columns = [c.replace("close_", "ret_") for c in log_returns.columns]

    n = len(log_returns)
    test_n = int(n * TEST_SIZE)
    train_n = n - test_n
    in_sample = log_returns.iloc[:train_n].copy()
    out_sample = log_returns.iloc[train_n:].copy()
    return in_sample, out_sample


def run_walkforward(walk_data, weight_return, weight_risk, all_weights):
    """Faithful replica of walkForwardOptimization3's loop (GA(100,50) per
    window, pick max-cost solution, compute test-period returns) that ALSO
    captures the winning weight vector per window.

    Returns (daily_return_series, list_of_weight_vectors).
    """
    periods = walk_forward_split(walk_data, [TRAIN_PERIOD, TEST_PERIOD])
    all_dates, all_returns, window_weights = [], [], []

    for p in periods:
        # Silence the GA's verbose per-generation prints.
        with contextlib.redirect_stdout(io.StringIO()):
            gen_dict = GA(100, 50, p[0], all_weights, weight_return, weight_risk)
        best_idx = gen_dict["best_cost"].index(max(gen_dict["best_cost"]))
        best_solution = np.array(gen_dict["best_param"][best_idx], dtype=float)

        returns = calc_returnOfPoints(p[1], best_solution)
        all_dates.extend(p[1].index.tolist())
        all_returns.extend(returns.tolist())
        window_weights.append(best_solution)

    series = pd.Series(all_returns, index=pd.DatetimeIndex(all_dates), name="Return")
    return series, window_weights


def perf_metrics(r):
    """Out-of-sample performance metrics, matching generate_final_results."""
    mean = r.mean()
    std = r.std()                       # ddof=1, same as the model's pandas .std()
    sharpe = mean / std                 # rf = 0
    cum_ret_pct = (np.exp(r.sum()) - 1) * 100
    cumulative = np.exp(r.cumsum())
    running_max = cumulative.expanding().max()
    mdd_pct = ((cumulative - running_max) / running_max).min() * 100
    return mean, std, sharpe, cum_ret_pct, mdd_pct


def class_allocations(window_weights):
    """Average portfolio weight per asset class, averaged over walk-forward
    windows."""
    W = np.array(window_weights)                # shape (n_windows, 8)
    equity = W[:, EQUITY_IDX].sum(axis=1).mean()
    gold = W[:, GOLD_IDX].sum(axis=1).mean()
    fixed = W[:, FIXED_INCOME_IDX].sum(axis=1).mean()
    return equity, gold, fixed


def main():
    print("=" * 100)
    print("ALPHA/BETA SENSITIVITY ANALYSIS  (fallback b: fixed outer window, inner GA only)")
    print("=" * 100)
    print(f"Fixed walk-forward window : train={TRAIN_PERIOD}, test={TEST_PERIOD} trading days")
    print(f"Alpha grid                : {ALPHAS}  (beta = 1 - alpha)")
    print(f"Seeds                     : SEED_A={SEED_A}, SEED_B={SEED_B} (numpy + random both seeded)")

    in_sample, out_sample = load_split()
    walk_data = pd.concat([in_sample.tail(TRAIN_PERIOD), out_sample], axis=0)

    print(f"Log-return rows           : {len(in_sample) + len(out_sample)} "
          f"(in-sample {len(in_sample)}, out-of-sample {len(out_sample)})")

    print("\nBuilding valid weight grid (make_weights)...")
    all_weights = make_weights()
    print(f"Valid weight vectors      : {len(all_weights)}")

    rows = []
    n_test_days = None
    date_range = None

    for alpha in ALPHAS:
        beta = round(1 - alpha, 10)
        for seed in (SEED_A, SEED_B):
            np.random.seed(seed)
            random.seed(seed)

            series, window_weights = run_walkforward(walk_data, alpha, beta, all_weights)
            mean, std, sharpe, cum, mdd = perf_metrics(series)
            eq, gold, fixed = class_allocations(window_weights)

            if n_test_days is None:
                n_test_days = len(series)
                date_range = (series.index.min().date(), series.index.max().date())

            rows.append({
                "alpha": alpha,
                "beta": beta,
                "seed": seed,
                "mean_daily_return": round(mean, 4),
                "std_daily_return": round(std, 4),
                "sharpe_rf0": round(sharpe, 4),
                "cumulative_return_pct": round(cum, 4),
                "max_drawdown_pct": round(mdd, 4),
                "avg_wt_equity": round(eq, 4),
                "avg_wt_gold": round(gold, 4),
                "avg_wt_fixed_income": round(fixed, 4),
            })
            print(f"  done: alpha={alpha:.1f} beta={beta:.1f} seed={seed:<2}  "
                  f"std={std:.4f}  MDD%={mdd:.4f}  eq={eq:.3f} fi={fixed:.3f}")

    df = pd.DataFrame(rows).sort_values(["alpha", "seed"]).reset_index(drop=True)

    print(f"\nOut-of-sample test days per run : {n_test_days}")
    print(f"Out-of-sample date range        : {date_range[0]} -> {date_range[1]}")
    print("(fixed 70/40 window => first 160 of 196 OOS days; window begins at the first OOS day)")

    # ---------------- full per-seed table ----------------
    print("\n" + "=" * 100)
    print("PER-SEED RESULTS (sorted by alpha)")
    print("=" * 100)
    print(df.to_string(index=False))

    # ---------------- seed-averaged summary (monotonicity view) ----------------
    agg = df.groupby("alpha").agg(
        beta=("beta", "first"),
        mean_daily_return=("mean_daily_return", "mean"),
        std_daily_return=("std_daily_return", "mean"),
        sharpe_rf0=("sharpe_rf0", "mean"),
        cumulative_return_pct=("cumulative_return_pct", "mean"),
        max_drawdown_pct=("max_drawdown_pct", "mean"),
        avg_wt_equity=("avg_wt_equity", "mean"),
        avg_wt_gold=("avg_wt_gold", "mean"),
        avg_wt_fixed_income=("avg_wt_fixed_income", "mean"),
    ).round(4).reset_index()

    print("\n" + "=" * 100)
    print("SEED-AVERAGED SUMMARY (the risk-dial view)")
    print("=" * 100)
    print(agg.to_string(index=False))

    df.to_csv(OUT_CSV, index=False)
    print(f"\nCSV written: {OUT_CSV}")


if __name__ == "__main__":
    main()
