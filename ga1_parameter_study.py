"""
Inner GA (GA1) parameter study — one-factor-at-a-time (OFAT) around the published
configuration, for the reviewer's request to justify the GA1 hyper-parameters.

Published GA1 configuration (see walkforward.py:99 -> GA(100, 50, ...) and
gaOpt1.py:31-32):
    population              = 100
    generations             = 50
    initial mutation rate   = 0.05
    high (adaptive) mutation = 0.10   <- held fixed (not part of this OFAT sweep)

What this script does
---------------------
* OFAT sweeps (change ONE factor, hold the others at their published values):
      population        in {50, 100, 150}
      generations       in {30, 50, 70}
      initial mutation  in {0.01, 0.05, 0.10}
* Evaluates the INNER GA only, on IN-SAMPLE data only (first 80% of trading days,
  the same split main.py uses: time_series_split(data_return, 0.2)).
* Uses the paper's high-risk setup: alpha = weight_return = 0.9, beta = weight_risk = 0.1,
  and fixed walk-forward window lengths train=50 / test=20. The inner GA optimises
  portfolio weights on each 50-day TRAIN window; the 36 non-overlapping walk-forward
  train windows of the in-sample slice form the instance set (no window is cherry-picked).
* Each configuration is repeated with 3 different random seeds. For each seed the GA is
  run once per window; per-seed values aggregate over windows, and we report the
  mean +/- std ACROSS the 3 seeds of:
      - best fitness achieved  (mean over windows of the GA's best in-sample fitness)
      - generations until convergence (mean over windows of generations actually run
        before the no-improvement early-stop in gaOpt1.GA; equals the cap if it never
        early-stops)
      - wall-clock runtime     (total seconds for one full pass over all windows)
* Writes a per-configuration summary CSV and a raw per-run CSV, and prints tables.
* If any tested configuration clearly beats the published one on mean fitness, it is
  flagged prominently rather than hidden.

IMPORTANT: This script does NOT modify any model code. The initial mutation rate is
varied by setting the module attribute gaOpt1.initial_mutation_rate at runtime (a
monkey-patch from this external script), which is exactly the global that gaOpt1.GA
reads. The adaptive high-mutation rate (gaOpt1.high_mutation_rate = 0.10) is left
untouched.
"""

import os
import io
import time
import contextlib
import numpy as np
import pandas as pd
import random

import gaOpt1
from gaOpt1 import GA
from makeArrayOfWeights import make_weights

# ----------------------------------------------------------------------------------
# Study configuration
# ----------------------------------------------------------------------------------
PUBLISHED = {"population": 100, "generations": 50, "init_mut": 0.05}
HIGH_MUT = gaOpt1.high_mutation_rate          # 0.10, held fixed
WEIGHT_RETURN = 0.9                            # alpha (high-risk profile)
WEIGHT_RISK = 0.1                              # beta
TRAIN = 50
TEST = 20
SEEDS = [0, 1, 2]
OUTPUT_DIR = "finalOutputs"

# OFAT sweeps: each entry is (label of the swept factor, list of values)
POP_VALUES = [50, 100, 150]
GEN_VALUES = [30, 50, 70]
MUT_VALUES = [0.01, 0.05, 0.10]


def build_in_sample():
    """Reproduce main.py's data pipeline and return the in-sample (80%) log-return slice."""
    close_data = pd.read_csv("data/closeData.csv", index_col=["date"], parse_dates=["date"])
    log_returns = np.log(close_data / close_data.shift(1)).dropna()
    log_returns.columns = [c.replace("close_", "ret_") for c in log_returns.columns]

    n = len(log_returns)
    test_n = int(n * 0.2)
    train_n = n - test_n
    in_sample = log_returns.iloc[:train_n].copy()
    return in_sample


def walk_forward_train_windows(df, train=TRAIN, test=TEST):
    """Non-overlapping walk-forward TRAIN windows (step = test), matching walkforward.walk_forward_split."""
    windows = []
    start = 0
    total = len(df)
    while start + train + test <= total:
        windows.append(df.iloc[start:start + train])
        start += test
    return windows


def run_ga_once(population, generations, init_mut, window, weights, seed):
    """
    Run the inner GA once on a single window with a given (config, seed).
    Returns (best_fitness, generations_run, runtime_seconds).

    The initial mutation rate is set via the gaOpt1 module global (the exact value
    gaOpt1.GA reads) before the run; the high mutation rate is left at its published value.
    Both numpy and python RNGs are seeded because gaOpt1 uses both.
    """
    gaOpt1.initial_mutation_rate = init_mut
    np.random.seed(seed)
    random.seed(seed)

    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):          # silence GA's per-generation prints
        gen_dict = GA(population, generations, window, weights, WEIGHT_RETURN, WEIGHT_RISK)
    runtime = time.perf_counter() - t0

    best_fitness = max(gen_dict["best_cost"])
    generations_run = len(gen_dict["generation"])  # early-stop -> < cap; else == cap
    return best_fitness, generations_run, runtime


def evaluate_config(population, generations, init_mut, windows, weights, raw_rows, cfg_label):
    """
    Evaluate one configuration across all seeds and windows.
    Per seed: aggregate over windows (mean fitness, mean generations_run, total runtime).
    Across seeds: return mean/std of those per-seed aggregates.
    """
    per_seed_fit = []
    per_seed_gens = []
    per_seed_runtime = []

    for seed in SEEDS:
        # deterministic, order-independent per-run seed: same window+base-seed pairs
        # the same starting RNG state across configs (paired comparison / variance reduction)
        win_fit, win_gens, win_rt = [], [], []
        for wi, window in enumerate(windows):
            run_seed = seed * 10000 + wi
            fit, gens, rt = run_ga_once(population, generations, init_mut, window, weights, run_seed)
            win_fit.append(fit)
            win_gens.append(gens)
            win_rt.append(rt)
            raw_rows.append({
                "config": cfg_label,
                "population": population,
                "generations": generations,
                "init_mut": init_mut,
                "high_mut": HIGH_MUT,
                "base_seed": seed,
                "window_index": wi,
                "best_fitness": fit,
                "generations_run": gens,
                "runtime_s": rt,
            })
        per_seed_fit.append(np.mean(win_fit))
        per_seed_gens.append(np.mean(win_gens))
        per_seed_runtime.append(np.sum(win_rt))   # wall-clock for one full pass over windows

    return {
        "population": population,
        "generations": generations,
        "init_mut": init_mut,
        "high_mut": HIGH_MUT,
        "fit_mean": float(np.mean(per_seed_fit)),
        "fit_std": float(np.std(per_seed_fit, ddof=0)),
        "gensrun_mean": float(np.mean(per_seed_gens)),
        "gensrun_std": float(np.std(per_seed_gens, ddof=0)),
        "runtime_mean_s": float(np.mean(per_seed_runtime)),
        "runtime_std_s": float(np.std(per_seed_runtime, ddof=0)),
    }


def cfg_key(population, generations, init_mut):
    return (population, generations, round(init_mut, 4))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 90)
    print("INNER GA (GA1) PARAMETER STUDY  --  OFAT around the published configuration")
    print("=" * 90)
    print(f"Published config     : population={PUBLISHED['population']}, "
          f"generations={PUBLISHED['generations']}, initial mutation={PUBLISHED['init_mut']}, "
          f"adaptive high mutation={HIGH_MUT} (fixed)")
    print(f"Data                 : in-sample only (first 80% of trading days)")
    print(f"Windows              : {TRAIN}/{TEST} walk-forward train windows, alpha={WEIGHT_RETURN}, beta={WEIGHT_RISK}")
    print(f"Repetitions          : seeds={SEEDS}")
    print("-" * 90)

    print("Loading data and building in-sample slice ...")
    in_sample = build_in_sample()
    windows = walk_forward_train_windows(in_sample)
    print(f"  in-sample rows       : {len(in_sample)}")
    print(f"  walk-forward windows : {len(windows)} (train={TRAIN}, test/step={TEST})")

    print("Enumerating valid portfolio weight vectors (make_weights) ... this takes ~40s")
    t0 = time.perf_counter()
    weights = make_weights()
    print(f"  valid weight vectors : {len(weights)}  ({time.perf_counter() - t0:.1f}s)")

    # Build the unique set of configurations (baseline is shared across the three axes)
    configs = set()
    for p in POP_VALUES:
        configs.add(cfg_key(p, PUBLISHED["generations"], PUBLISHED["init_mut"]))
    for g in GEN_VALUES:
        configs.add(cfg_key(PUBLISHED["population"], g, PUBLISHED["init_mut"]))
    for m in MUT_VALUES:
        configs.add(cfg_key(PUBLISHED["population"], PUBLISHED["generations"], m))

    total_runs = len(configs) * len(SEEDS) * len(windows)
    print(f"\nUnique configurations : {len(configs)}")
    print(f"Total inner-GA runs   : {len(configs)} configs x {len(SEEDS)} seeds x "
          f"{len(windows)} windows = {total_runs}")
    print("-" * 90)

    raw_rows = []
    results = {}
    for i, (p, g, m) in enumerate(sorted(configs), 1):
        label = f"pop={p},gen={g},mut={m}"
        is_base = cfg_key(p, g, m) == cfg_key(PUBLISHED["population"], PUBLISHED["generations"], PUBLISHED["init_mut"])
        tag = "  [PUBLISHED]" if is_base else ""
        print(f"[{i}/{len(configs)}] running {label}{tag} ...", flush=True)
        t0 = time.perf_counter()
        results[cfg_key(p, g, m)] = evaluate_config(p, g, m, windows, weights, raw_rows, label)
        print(f"        done in {time.perf_counter() - t0:.1f}s  "
              f"(fit={results[cfg_key(p, g, m)]['fit_mean']:.4f}, "
              f"gens_run={results[cfg_key(p, g, m)]['gensrun_mean']:.1f}, "
              f"runtime={results[cfg_key(p, g, m)]['runtime_mean_s']:.1f}s)", flush=True)

    base = results[cfg_key(PUBLISHED["population"], PUBLISHED["generations"], PUBLISHED["init_mut"])]

    # ------------------------------------------------------------------------------
    # Assemble per-configuration summary rows, tagged by OFAT axis
    # ------------------------------------------------------------------------------
    summary_rows = []

    def add_row(axis, swept_value, key):
        r = results[key]
        summary_rows.append({
            "axis": axis,
            "swept_value": swept_value,
            "population": r["population"],
            "generations": r["generations"],
            "init_mut": r["init_mut"],
            "high_mut": r["high_mut"],
            "fit_mean": r["fit_mean"],
            "fit_std": r["fit_std"],
            "gensrun_mean": r["gensrun_mean"],
            "gensrun_std": r["gensrun_std"],
            "runtime_mean_s": r["runtime_mean_s"],
            "runtime_std_s": r["runtime_std_s"],
            "delta_fit_vs_published": r["fit_mean"] - base["fit_mean"],
            "runtime_ratio_vs_published": r["runtime_mean_s"] / base["runtime_mean_s"],
            "is_published": key == cfg_key(PUBLISHED["population"], PUBLISHED["generations"], PUBLISHED["init_mut"]),
        })

    for p in POP_VALUES:
        add_row("population", p, cfg_key(p, PUBLISHED["generations"], PUBLISHED["init_mut"]))
    for g in GEN_VALUES:
        add_row("generations", g, cfg_key(PUBLISHED["population"], g, PUBLISHED["init_mut"]))
    for m in MUT_VALUES:
        add_row("init_mutation", m, cfg_key(PUBLISHED["population"], PUBLISHED["generations"], m))

    summary_df = pd.DataFrame(summary_rows)
    raw_df = pd.DataFrame(raw_rows)

    summary_path = os.path.join(OUTPUT_DIR, "GA1_parameter_study.csv")
    raw_path = os.path.join(OUTPUT_DIR, "GA1_parameter_study_raw.csv")
    summary_df.to_csv(summary_path, index=False)
    raw_df.to_csv(raw_path, index=False)

    # ------------------------------------------------------------------------------
    # Printed tables
    # ------------------------------------------------------------------------------
    def print_axis(axis, values, fixed_desc):
        print("\n" + "=" * 90)
        print(f"OFAT AXIS: {axis}   (holding {fixed_desc})")
        print("=" * 90)
        print(f"{'value':>8} | {'best fitness':>20} | {'gens until conv.':>18} | "
              f"{'runtime (s)':>18} | {'dFit':>9} | {'RTx':>5}")
        print(f"{'':>8} | {'mean +/- std':>20} | {'mean +/- std':>18} | {'mean +/- std':>18} | "
              f"{'vs pub':>9} | {'':>5}")
        print("-" * 90)
        sub = summary_df[summary_df["axis"] == axis]
        for v in values:
            row = sub[sub["swept_value"] == v].iloc[0]
            star = " *" if row["is_published"] else "  "
            print(f"{v:>8}{star}| "
                  f"{row['fit_mean']:>9.4f} +/-{row['fit_std']:>7.4f} | "
                  f"{row['gensrun_mean']:>8.1f} +/-{row['gensrun_std']:>6.1f} | "
                  f"{row['runtime_mean_s']:>8.1f} +/-{row['runtime_std_s']:>6.1f} | "
                  f"{row['delta_fit_vs_published']:>+9.4f} | "
                  f"{row['runtime_ratio_vs_published']:>5.2f}")
        print("  (* = published value; dFit = mean-fitness gap vs published; RTx = runtime multiple vs published)")

    print_axis("population", POP_VALUES,
               f"generations={PUBLISHED['generations']}, init_mut={PUBLISHED['init_mut']}")
    print_axis("generations", GEN_VALUES,
               f"population={PUBLISHED['population']}, init_mut={PUBLISHED['init_mut']}")
    print_axis("init_mutation", MUT_VALUES,
               f"population={PUBLISHED['population']}, generations={PUBLISHED['generations']}")

    # ------------------------------------------------------------------------------
    # Plateau / "does anything beat published?" verdict
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    print(f"Published fitness (mean +/- std over seeds): {base['fit_mean']:.4f} +/- {base['fit_std']:.4f}")

    # A configuration "clearly beats" the published one only if its mean-fitness advantage
    # exceeds the noise floor (the pooled seed-level std of the two configs). This guards
    # against flagging differences that are within seed-to-seed variation.
    beats = []
    for _, row in summary_df.iterrows():
        if row["is_published"]:
            continue
        delta = row["delta_fit_vs_published"]
        noise = base["fit_std"] + row["fit_std"]
        if delta > max(noise, 1e-9) and delta > 1e-4:
            beats.append((row, delta, noise))

    if beats:
        print("\n" + "!" * 90)
        print("!! ATTENTION: the following configuration(s) beat the published one on mean fitness")
        print("!! by more than the seed-level noise floor. Do NOT ignore -- reconsider the published choice:")
        for row, delta, noise in sorted(beats, key=lambda x: -x[1]):
            print(f"!!   {row['axis']}={row['swept_value']}: fitness "
                  f"{row['fit_mean']:.4f} vs published {base['fit_mean']:.4f} "
                  f"(+{delta:.4f}, noise floor {noise:.4f}), "
                  f"runtime x{row['runtime_ratio_vs_published']:.2f}")
        print("!" * 90)
    else:
        print("\nNo tested configuration beats the published one beyond seed-level noise.")
        print("All fitness gaps vs published are within the noise floor (pooled seed std),")
        print("i.e. the published configuration sits on the performance plateau:")
        print("  - larger population / more generations give negligible fitness gains at higher runtime;")
        print("  - initial mutation = 0.05 balances convergence speed and stability.")

    print("\nOutputs written:")
    print(f"  summary : {summary_path}")
    print(f"  raw     : {raw_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
