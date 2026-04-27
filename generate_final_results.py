"""
Generate Final Results - Complete Evaluation Pipeline

This script generates all final outputs including:
- Performance metrics (Results_metrics.csv)
- Cumulative return charts
- Drawdown charts
- Statistical test results

All outputs are saved to the finalOutputs folder.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, mannwhitneyu
from markowitz_train_test import markowitz_train_test_project
from equal_weight_train_test import equal_weight_train_test_project
import os

def calculate_metrics(returns):
    """Calculate performance metrics for all portfolios"""
    metrics = pd.DataFrame(index=returns.columns)
    metrics['Mean Daily Log Return'] = returns.mean()
    metrics['Standard Deviation'] = returns.std()
    metrics['Sharpe Ratio'] = metrics['Mean Daily Log Return'] / metrics['Standard Deviation']
    metrics['Cumulative Return %'] = (np.exp(returns.sum()) - 1) * 100
    
    # Calculate Maximum Drawdown (MDD)
    cumulative = np.exp(returns.cumsum())
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics['MDD'] = drawdown.min()
    
    return metrics, drawdown

def plot_cumulative_returns(returns, output_dir, portfolios=['highRisk', 'lowRisk', 'normalRisk']):
    """Generate cumulative return charts"""
    cumulative = np.exp(returns.cumsum())
    
    # Individual portfolio charts
    for portfolio in portfolios:
        if portfolio in returns.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(cumulative.index, cumulative[portfolio], label=portfolio, linewidth=2)
            # For normalRisk, also show benchmarks (agas, close_overal)
            if portfolio == 'normalRisk':
                for bench in ['agas', 'close_overal', 'markowitz', 'equalweight']:
                    if bench in cumulative.columns:
                        plt.plot(cumulative.index, cumulative[bench], label=bench, linewidth=2)
            plt.title(f'Cumulative Return - {portfolio}', fontsize=14, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Return', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = os.path.join(output_dir, f'Cum_{portfolio}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {filename}")
    
    # Combined portfolio chart
    plt.figure(figsize=(12, 6))
    for portfolio in portfolios:
        if portfolio in returns.columns:
            plt.plot(cumulative.index, cumulative[portfolio], label=portfolio, linewidth=2)
    plt.title('Cumulative Return - All Portfolios', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = os.path.join(output_dir, 'Cum_portfolioS.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

    # Combined portfolio chart & benchmarks
    plt.figure(figsize=(12, 6))
    for portfolio in portfolios + ['agas', 'close_overal', 'markowitz', 'equalweight']:
        if portfolio in returns.columns:
            plt.plot(cumulative.index, cumulative[portfolio], label=portfolio, linewidth=2)
    plt.title('Cumulative Return - All Portfolios & Benchmarks', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = os.path.join(output_dir, 'Cum_portfolioS-benchmarkS.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def plot_drawdowns(drawdowns, output_dir, portfolios=['highRisk', 'lowRisk', 'normalRisk']):
    """Generate drawdown charts"""
    # Individual drawdown chart for normalRisk (with benchmarks agas, close_overal)
    if 'normalRisk' in drawdowns.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(drawdowns.index, drawdowns['normalRisk'], label='normalRisk', linewidth=2)
        for bench in ['agas', 'close_overal', 'markowitz', 'equalweight']:
            if bench in drawdowns.columns:
                plt.plot(drawdowns.index, drawdowns[bench], label=bench, linewidth=2)
        plt.title('Drawdown Chart - normalRisk', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = os.path.join(output_dir, 'DD_normalRisk.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # Combined drawdown chart
    plt.figure(figsize=(12, 6))
    for portfolio in portfolios:
        if portfolio in drawdowns.columns:
            plt.plot(drawdowns.index, drawdowns[portfolio], label=portfolio, linewidth=2)
    plt.title('Drawdown Chart - All Portfolios', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = os.path.join(output_dir, 'DD_portfolioS.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

    # Combined drawdown chart2
    plt.figure(figsize=(12, 6))
    for portfolio in portfolios + ['agas', 'close_overal', 'markowitz', 'equalweight']:
        if portfolio in drawdowns.columns:
            plt.plot(drawdowns.index, drawdowns[portfolio], label=portfolio, linewidth=2)
    plt.title('Drawdown Chart - All Portfolios & Benchmarks', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = os.path.join(output_dir, 'DD_portfolioS-benchmarkS.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def perform_statistical_tests(returns, output_dir):
    """Perform all statistical tests and save results"""
    test_results = []
    
    # Shapiro-Wilk Test (Normality)
    print("\n" + "="*80)
    print("Shapiro-Wilk Test (Normality)")
    print("="*80)
    for col in returns.columns:
        stat, p_value = shapiro(returns[col])
        test_results.append({
            'Test': 'Shapiro-Wilk',
            'Portfolio': col,
            'Statistic': stat,
            'P-value': p_value,
            'Interpretation': 'Normal' if p_value > 0.05 else 'Not Normal'
        })
        print(f"Shapiro-Wilk Test ({col}): {stat:.6f} {p_value:.2e}")
    
    # Levene Test (Variance Equality)
    print("\n" + "="*80)
    print("Levene Test (Variance Equality)")
    print("="*80)
    test_pairs = [
        ('agas', 'normalRisk'),
        ('highRisk', 'normalRisk'),
        ('lowRisk', 'normalRisk'),
        ('highRisk', 'lowRisk'),
        ('highRisk', 'markowitz'),
        ('lowRisk', 'markowitz'),
        ('normalRisk', 'markowitz'),
        ('highRisk', 'equalweight'),
        ('lowRisk', 'equalweight'),
        ('normalRisk', 'equalweight')
    ]
    
    for pair in test_pairs:
        if pair[0] in returns.columns and pair[1] in returns.columns:
            stat, p_value = levene(returns[pair[0]], returns[pair[1]])
            test_results.append({
                'Test': 'Levene',
                'Portfolio': f"{pair[0]} vs {pair[1]}",
                'Statistic': stat,
                'P-value': p_value,
                'Interpretation': 'Equal Variances' if p_value > 0.05 else 'Different Variances'
            })
            print(f"Levene({pair[0]}, {pair[1]}): {stat:.6f} {p_value:.2e}")
    
    # Mann-Whitney U Test
    print("\n" + "="*80)
    print("Mann-Whitney U Test")
    print("="*80)
    mw_pairs = [
        ('agas', 'normalRisk'),
        ('close_overal', 'normalRisk'),
        ('highRisk', 'normalRisk'),
        ('lowRisk', 'highRisk'),
        ('lowRisk', 'normalRisk'),
        ('highRisk', 'markowitz'),
        ('lowRisk', 'markowitz'),
        ('normalRisk', 'markowitz'),
        ('highRisk', 'equalweight'),
        ('lowRisk', 'equalweight'),
        ('normalRisk', 'equalweight')
    ]
    
    for pair in mw_pairs:
        if pair[0] in returns.columns and pair[1] in returns.columns:
            u_stat, p_value = mannwhitneyu(
                returns[pair[0]],
                returns[pair[1]],
                alternative='two-sided'
            )
            test_results.append({
                'Test': 'Mann-Whitney U',
                'Portfolio': f"{pair[0]} vs {pair[1]}",
                'Statistic': u_stat,
                'P-value': p_value,
                'Interpretation': 'No significant difference' if p_value > 0.05 else 'Significant difference'
            })
            print(f"Mann-Whitney U ({pair[0]} vs {pair[1]}): {u_stat:.2f} {p_value:.6f}")
    
    # Save test results to CSV
    test_df = pd.DataFrame(test_results)
    test_file = os.path.join(output_dir, 'Statistical_Tests_Results.csv')
    test_df.to_csv(test_file, index=False)
    print(f"\nStatistical test results saved to: {test_file}")
    
    return test_df

def generate_final_results(insample_returns, outsample_returns, output_dir="finalOutputs", benchmark_file="data/logRetBenchETF.csv"):
    """
    Generate all final results including metrics, charts, and statistical tests.
    
    Parameters:
    -----------
    output_dir : str
        Directory where all outputs will be saved
    benchmark_file : str
        Path to benchmark ETF returns file
    """
    print("="*80)
    print("GENERATING FINAL RESULTS")
    print("="*80)
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Load benchmark data
    print("\nLoading benchmark data...")
    df_index = pd.read_csv(benchmark_file)
    df_index['date'] = pd.to_datetime(df_index['date'], format='%m/%d/%Y')
    df_index = df_index.set_index('date')
    df_index = df_index[["agas", "close_overal"]]
    df_index.index.name = "Date"
    print(f"Benchmark data loaded: {len(df_index)} days")
    
    # markowitzModel & equalWeightModel

    data_return_markowitz = markowitz_train_test_project(insample_returns, outsample_returns, save_test_returns_csv=True)
    data_return_equalweight = equal_weight_train_test_project(insample_returns, outsample_returns, save_test_returns_csv=True)

    # Load portfolio return files
    print("\nLoading portfolio return files...")
    portfolio_files = {
        'highRisk': os.path.join(output_dir, 'dailyReturn_highRisk.csv'),
        'lowRisk': os.path.join(output_dir, 'dailyReturn_lowRisk.csv'),
        'normalRisk': os.path.join(output_dir, 'dailyReturn_normalRisk.csv'),
        'equalweight': os.path.join(output_dir, 'equal_weight_test_returns.csv'),
        'markowitz': os.path.join(output_dir, 'markowitz_test_returns.csv')
    }
    
    dfs = {}
    for key, filepath in portfolio_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=["Date"], index_col=["Date"])
            # Remove Unnamed: 0 column if exists
            if 'Unnamed: 0' in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)
            # Rename Return column
            df = df.rename(columns={"Return": key})
            dfs[key] = df
            print(f"  Loaded: {filepath} ({len(df)} days)")
        else:
            print(f"  Warning: {filepath} not found!")
    
    # Merge all dataframes
    print("\nMerging data...")
    df_merged = dfs['highRisk'].join([dfs['lowRisk'], dfs['normalRisk'], df_index, dfs['equalweight'], dfs['markowitz']], how='inner')
    returns = df_merged.copy().dropna()
    print(f"Merged data: {len(returns)} days, {len(returns.columns)} portfolios/benchmarks")
    
    # Calculate metrics
    print("\n" + "="*80)
    print("Calculating Performance Metrics")
    print("="*80)
    metrics, drawdowns = calculate_metrics(returns)
    
    # Save metrics to CSV
    metrics_file = os.path.join(output_dir, 'Results_metrics.csv')
    metrics.to_csv(metrics_file)
    print(f"\nMetrics saved to: {metrics_file}")
    print("\nPerformance Metrics:")
    print(metrics.round(6))
    
    # Generate cumulative return charts
    print("\n" + "="*80)
    print("Generating Cumulative Return Charts")
    print("="*80)
    plot_cumulative_returns(returns, output_dir)
    
    # Generate drawdown charts
    print("\n" + "="*80)
    print("Generating Drawdown Charts")
    print("="*80)
    plot_drawdowns(drawdowns, output_dir)
    
    # Perform statistical tests
    print("\n" + "="*80)
    print("Performing Statistical Tests")
    print("="*80)
    test_results = perform_statistical_tests(returns, output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("ALL RESULTS GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - Results_metrics.csv")
    print(f"  - Statistical_Tests_Results.csv")
    print(f"  - Cum_highRisk.png")
    print(f"  - Cum_lowRisk.png")
    print(f"  - Cum_normalRisk.png")
    print(f"  - Cum_portfolioS.png")
    print(f"  - DD_normalRisk.png")
    print(f"  - DD_portfolioS.png")
    print("="*80)

if __name__ == '__main__':
    # Load close price data and calculate log returns
    close_data = pd.read_csv("data/closeData.csv", index_col=["date"], parse_dates=["date"])
    # Calculate log returns
    log_returns = np.log(close_data / close_data.shift(1)).dropna()
    # Rename columns from 'close_*' to 'ret_*'
    log_returns.columns = [col.replace('close_', 'ret_') for col in log_returns.columns]
    # Use log returns as data_return for optimization
    data_return = log_returns

    # separation of in sample and out sample data
    def time_series_split(df, test_size=0.2):
        n = len(df)
        test_n = int(n * test_size)
        train_n = n - test_n
        
        train = df.iloc[:train_n].copy()
        test  = df.iloc[train_n:].copy()
        
        return train, test

    inSample_data, outSample_data = time_series_split(data_return, 0.2)

    generate_final_results(insample_returns=inSample_data, outsample_returns=outSample_data)
