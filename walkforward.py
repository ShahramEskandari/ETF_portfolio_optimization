import pandas as pd
import numpy as np
from gaOpt1 import *
from makeArrayOfWeights import *
from fitness_ga1 import calc_riskAdjRet

benchmarkRetPoint = 0.0008
## section: working with just GA1
# data_return = pd.read_csv("logRetData.csv", index_col=["date"], parse_dates=["date"])
# allPossibleWeights = make_weights()

# gen_dict = GA(100,30,data_return.tail(20), allPossibleWeights)
# cost_gen = max(gen_dict["best_cost"])
# best_index_in_generation = gen_dict["best_cost"].index(max(gen_dict["best_cost"]))
# bestSolution = gen_dict["best_param"][best_index_in_generation]

# print("best param = ", bestSolution)
# print("best cost = ", cost_gen)
# print(np.sum(bestSolution))

def walk_forward_split(df, arr_train_test):
    """
    Generate walk-forward train-test splits from a DataFrame.

    Parameters:
    - df: the DataFrame (assumes rows are time-ordered)
    - train_period: number of rows for training
    - test_period: number of rows for testing

    Returns:
    - List of (train_df, test_df) tuples
    """
    train_period = int(arr_train_test[0])
    test_period = int(arr_train_test[1])
    splits = []
    start = 0
    total_rows = len(df)

    while start + train_period + test_period <= total_rows:
        train_df = df.iloc[int(start) : int(start + train_period)]
        test_df = df.iloc[int(start + train_period) : int(start + train_period + test_period)]
        splits.append((train_df, test_df))
        start += test_period  # Move forward by test window (non-overlapping test sets)

    return splits

def walkForwardOptimization(dataframe, train_test_array, allPossibleWeights):
    periods = walk_forward_split(dataframe, train_test_array)

    metics = []
    for p in periods:
        gen_dict = GA(70, 40, p[0], allPossibleWeights)
        cost_gen = max(gen_dict["best_cost"])
        best_index_in_generation = gen_dict["best_cost"].index(max(gen_dict["best_cost"]))
        bestSolution = gen_dict["best_param"][best_index_in_generation]

        metics.append(calc_riskAdjRet(p[1], bestSolution))

    metics = np.array(metics)
    # you can use any metrics
    fitness_value = metics.mean()

    return fitness_value

def walkForwardOptimization2(dataframe, train_test_array, allPossibleWeights):
    periods = walk_forward_split(dataframe, train_test_array)

    metics = []
    for p in periods:
        gen_dict = GA(200, 50, p[0], allPossibleWeights)
        cost_gen = max(gen_dict["best_cost"])
        best_index_in_generation = gen_dict["best_cost"].index(max(gen_dict["best_cost"]))
        bestSolution = gen_dict["best_param"][best_index_in_generation]

        # metics.append(calc_riskAdjRet(p[1], bestSolution))
        metics.append(calc_returnOfPoints(p[1], bestSolution))

    # metics = np.array(metics)
    # you can use any metrics like :
    merged_metrics = np.concatenate(metics)
    if merged_metrics.mean() > benchmarkRetPoint:
        fitness_value = 1/merged_metrics.std()
    else:
        fitness_value = -99
        
    #fitness_value = metics.mean()

    return fitness_value


def walkForwardOptimization3(dataframe, train_test_array, allPossibleWeights, create_csv=False, csv_name=None, weight_return=0.8, weight_risk=0.2):
    periods = walk_forward_split(dataframe, train_test_array)

    metrics = []
    all_dates = []
    all_returns = []
    
    for p in periods:
        gen_dict = GA(20, 5, p[0], allPossibleWeights, weight_return, weight_risk)
        cost_gen = max(gen_dict["best_cost"])
        best_index_in_generation = gen_dict["best_cost"].index(max(gen_dict["best_cost"]))
        bestSolution = gen_dict["best_param"][best_index_in_generation]

        # Calculate returns for the test period
        returns = calc_returnOfPoints(p[1], bestSolution)
        metrics.append(returns)
        
        # Store dates and returns for CSV creation
        if create_csv:
            all_dates.extend(p[1].index.tolist())
            all_returns.extend(returns.tolist())

    # Merge all metrics
    merged_metrics = np.concatenate(metrics)
        
    fitness_value = merged_metrics.mean()
    
    # Create and save CSV if requested
    if create_csv:
        if csv_name is None:
            csv_name = "daily_returns.csv"
        
        # Create DataFrame with dates as index and returns as column
        df_returns = pd.DataFrame({
            'Return': all_returns
        }, index=all_dates)
        
        # Save to CSV
        df_returns.to_csv(csv_name, index_label='Date')
        print(f"CSV file saved as: {csv_name}")

    return fitness_value

