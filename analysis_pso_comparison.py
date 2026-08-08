"""GA-vs-PSO comparison for the inner-layer portfolio optimizer.

The reviewer asked for a comparison of the model's optimization with a similar
nonlinear metaheuristic such as PSO. This standalone script benchmarks the
inner genetic algorithm against a standard global-best Particle Swarm
Optimization under IDENTICAL conditions:

  * same objective function (fitness_ga1.fitness_function with the same
    min-max extremums from findMinMax),
  * same constraints and repair rules (long-only, sum-to-one, 3-decimal
    rounding, >=12.5% per asset class via makeArrayOfWeights.is_valid),
  * same walk-forward protocol as the alpha/beta sensitivity study
    (fixed neutral outer window train=70/test=40, out-of-sample only),
  * same computational budget (population/swarm = 100, generations/iterations
    ceiling = 50) and the same two random seeds (42, 7),
  * both investor profiles: risk-seeking (alpha=0.9, beta=0.1) and
    risk-averse (alpha=0.2, beta=0.8).

PSO settings: constriction-style parameters w=0.72, c1=c2=1.49 (Clerc &
Kennedy, 2002). Velocity is applied in the raw 8-dimensional weight space and
each candidate is repaired exactly like a GA chromosome (clip at zero,
normalize, floor to 3 decimals, last gene absorbs the rounding residue); a
candidate that still violates the class floors keeps its previous position.

No model code is modified; this script only calls the existing functions.
Output: finalOutputs/PSO_vs_GA_Comparison.csv
"""

import os
import io
import sys
import time
import contextlib
import random
import numpy as np
import pandas as pd

from gaOpt1 import GA, generate_gene
from makeArrayOfWeights import make_weights, is_valid
from walkforward import walk_forward_split
from fitness_ga1 import fitness_function, findMinMax, calc_returnOfPoints

# ----------------------------- configuration -----------------------------
PROFILES = [
    {"name": "risk-seeking", "alpha": 0.9, "beta": 0.1},
    {"name": "risk-averse", "alpha": 0.2, "beta": 0.8},
]
SEEDS = [42, 7]
TRAIN_PERIOD = 70
TEST_PERIOD = 40
TEST_SIZE = 0.20
SWARM_SIZE = 100
MAX_ITER = 50
PSO_W, PSO_C1, PSO_C2 = 0.72, 1.49, 1.49
RF_ANNUAL = 0.20            # for the rf=20% Sharpe column
OUTPUT_DIR = "finalOutputs"
OUT_CSV = os.path.join(OUTPUT_DIR, "PSO_vs_GA_Comparison.csv")

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
    return log_returns.iloc[:train_n].copy(), log_returns.iloc[train_n:].copy()


def repair(w):
    """GA-style chromosome repair: clip, normalize, floor to 3 decimals,
    absorb residue in the last gene."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    s = w.sum()
    if s <= 0:
        return None
    w = w / s
    w = np.floor(w * 1000) / 1000
    w[-1] = round(1 - np.sum(w[:-1]), 3)
    if w[-1] < 0:
        return None
    return w


def pso_optimize(train_data, extermums, weight_return, weight_risk):
    """Global-best PSO over the constrained weight simplex; same budget as
    the inner GA (100 x 50). Returns the best repaired-valid weight vector."""
    positions = [generate_gene() for _ in range(SWARM_SIZE)]
    velocities = [np.random.uniform(-0.1, 0.1, 8) for _ in range(SWARM_SIZE)]

    def fit(w):
        return fitness_function(train_data, w, extermums, weight_return, weight_risk)

    pbest = [p.copy() for p in positions]
    pbest_cost = [fit(p) for p in positions]
    g_idx = int(np.argmax(pbest_cost))
    gbest, gbest_cost = pbest[g_idx].copy(), pbest_cost[g_idx]

    for _ in range(MAX_ITER):
        for i in range(SWARM_SIZE):
            r1, r2 = np.random.rand(8), np.random.rand(8)
            velocities[i] = (PSO_W * velocities[i]
                             + PSO_C1 * r1 * (pbest[i] - positions[i])
                             + PSO_C2 * r2 * (gbest - positions[i]))
            candidate = repair(positions[i] + velocities[i])
            if candidate is None or not is_valid(candidate):
                continue  # invalid move: keep previous position
            positions[i] = candidate
            c = fit(candidate)
            if c > pbest_cost[i]:
                pbest[i], pbest_cost[i] = candidate.copy(), c
                if c > gbest_cost:
                    gbest, gbest_cost = candidate.copy(), c
    return gbest, gbest_cost


def run_walkforward(walk_data, optimizer, weight_return, weight_risk, all_weights):
    """Walk-forward loop identical to walkForwardOptimization3; per window the
    chosen optimizer produces the weights applied to the untouched test slice."""
    periods = walk_forward_split(walk_data, [TRAIN_PERIOD, TEST_PERIOD])
    all_dates, all_returns, window_weights = [], [], []
    for p in periods:
        if optimizer == "GA":
            with contextlib.redirect_stdout(io.StringIO()):
                gen_dict = GA(100, 50, p[0], all_weights, weight_return, weight_risk)
            best_idx = gen_dict["best_cost"].index(max(gen_dict["best_cost"]))
            best = np.array(gen_dict["best_param"][best_idx], dtype=float)
        else:
            extermums = findMinMax(p[0], all_weights)
            best, _ = pso_optimize(p[0], extermums, weight_return, weight_risk)
        returns = calc_returnOfPoints(p[1], best)
        all_dates.extend(p[1].index.tolist())
        all_returns.extend(returns.tolist())
        window_weights.append(best)
    series = pd.Series(all_returns, index=pd.DatetimeIndex(all_dates), name="Return")
    return series, window_weights


def perf_metrics(r, rf_daily):
    mean = r.mean()
    std = r.std()
    cum_ret_pct = (np.exp(r.sum()) - 1) * 100
    cumulative = np.exp(r.cumsum())
    running_max = cumulative.expanding().max()
    mdd_pct = ((cumulative - running_max) / running_max).min() * 100
    return mean, std, mean / std, (mean - rf_daily) / std, cum_ret_pct, mdd_pct


def main():
    smoke = "--smoke" in sys.argv
    print("GA vs PSO comparison (fixed 70/40 window, out-of-sample)")
    in_sample, out_sample = load_split()
    walk_data = pd.concat([in_sample.tail(TRAIN_PERIOD), out_sample], axis=0)

    # trading days per year in the full sample, for the rf=20% Sharpe column
    full_index = pd.concat([in_sample, out_sample]).index
    years = (full_index[-1] - full_index[0]).days / 365.25
    days_per_year = len(full_index) / years
    rf_daily = np.log(1 + RF_ANNUAL) / days_per_year
    print(f"trading days/year = {days_per_year:.1f}  ->  daily log rf = {rf_daily:.6f}")

    t0 = time.time()
    print("building valid weight grid (make_weights)...")
    all_weights = make_weights()
    print(f"valid weight vectors: {len(all_weights)}  ({time.time()-t0:.0f}s)")

    global MAX_ITER
    profiles, seeds = PROFILES, SEEDS
    if smoke:
        profiles, seeds, MAX_ITER = PROFILES[:1], SEEDS[:1], 5
        print("SMOKE MODE: 1 profile, 1 seed, MAX_ITER=5")

    rows = []
    for prof in profiles:
        for optimizer in ("GA", "PSO"):
            for seed in seeds:
                np.random.seed(seed)
                random.seed(seed)
                t0 = time.time()
                series, ww = run_walkforward(walk_data, optimizer,
                                             prof["alpha"], prof["beta"], all_weights)
                mean, std, sh0, sh20, cum, mdd = perf_metrics(series, rf_daily)
                W = np.array(ww)
                rows.append({
                    "profile": prof["name"], "alpha": prof["alpha"], "beta": prof["beta"],
                    "optimizer": optimizer, "seed": seed,
                    "mean_daily_return": round(mean, 4), "std_daily_return": round(std, 4),
                    "sharpe_rf0": round(sh0, 4), "sharpe_rf20": round(sh20, 4),
                    "cumulative_return_pct": round(cum, 4), "max_drawdown_pct": round(mdd, 4),
                    "avg_wt_equity": round(W[:, EQUITY_IDX].sum(axis=1).mean(), 4),
                    "avg_wt_gold": round(W[:, GOLD_IDX].sum(axis=1).mean(), 4),
                    "avg_wt_fixed_income": round(W[:, FIXED_INCOME_IDX].sum(axis=1).mean(), 4),
                })
                print(f"  {prof['name']:<13} {optimizer:<4} seed={seed:<3} "
                      f"std={std:.4f} cum%={cum:.2f} mdd%={mdd:.2f} "
                      f"({time.time()-t0:.0f}s)")
                df = pd.DataFrame(rows)
                df.to_csv(OUT_CSV if not smoke else OUT_CSV + ".smoke", index=False)

    print("\nFINAL RESULTS")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nCSV written: {OUT_CSV if not smoke else OUT_CSV + '.smoke'}")


if __name__ == "__main__":
    main()
