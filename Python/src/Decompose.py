import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import seasonal_decompose

from UtilityPlot import plot_time_series


def apply_hp_filter_with_optimal_lamb(y, x, lamb_range):
    best_lamb = None
    best_trend = None
    best_cycle = None
    best_variance_ratio = float('inf')  # Initialize with a large value

    for lamb_ in lamb_range:
        cycle_, trend_ = hpfilter(y, lamb=lamb_)
        variance_ratio = np.var(cycle_) / np.var(y)

        # Check if the current result is better than the previous best
        if variance_ratio < best_variance_ratio:
            best_lamb = lamb_
            best_trend = trend_
            best_cycle = cycle_
            best_variance_ratio = variance_ratio

    # Plot the best result
    plt.figure(figsize=(10, 12))
    plt.subplot(3, 1, 1)
    plot_time_series(x, y, label='Original Time Series', title='Original Time Series')
    plt.subplot(3, 1, 2)
    plot_time_series(x, best_trend, label='Trend', title=f'Best Trend (lamb={best_lamb})', linestyle='--', color='red')
    plt.subplot(3, 1, 3)
    plot_time_series(x, best_cycle, label='Cycle', title=f'Best Cycle (lamb={best_lamb})', linestyle='--',
                     color='green')
    plt.tight_layout()
    plt.show()

    return best_cycle, best_trend, best_lamb


def seasonal_decomposition(y, x, period_range):
    best_period = None
    best_result = None
    best_seasonal_variance = float('-inf')  # Initialize with a small value
    best_residual_variance = float('inf')  # Initialize with a large value

    for period in period_range:
        result_ = seasonal_decompose(y, model='additive', period=period)
        seasonal_variance = np.var(result_.seasonal)
        residual_variance = np.var(result_.resid)

        # Check if the current result is better than the previous best
        if seasonal_variance > best_seasonal_variance and residual_variance < best_residual_variance:
            best_period = period
            best_result = result_
            best_seasonal_variance = seasonal_variance
            best_residual_variance = residual_variance

    print(f"Optimal period: {best_period}")

    # Plot the decomposition components
    plt.figure(figsize=(20, 20))
    # Plot Base Trend
    plt.subplot(4, 1, 1)
    plot_time_series(x, y, label='Base Trend', title='Base Trend', linestyle='--', color='red')
    # Plot Trend
    plt.subplot(4, 1, 2)
    plot_time_series(x, best_result.trend, label='Trend', title='Trend', linestyle='--', color='blue')
    # Plot Seasonal
    plt.subplot(4, 1, 3)
    plot_time_series(x, best_result.seasonal, label='Seasonal', title='Seasonal', linestyle='--', color='blue')
    # Plot Residual
    plt.subplot(4, 1, 4)
    plot_time_series(x, best_result.resid, label='Residual', title='Residual', linestyle='--', color='purple')
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()

    return best_result


# Create a DataFrame
df = pd.read_csv('generated_data.csv')
X, Y = df["X"], df["Y"]

cycle, trend, lamb = apply_hp_filter_with_optimal_lamb(y=df['Y'], x=df['X'], lamb_range=[1, 10, 50, 100, 200, 500, 0.5])
print(f"Optimal lamb: {lamb}")

result = seasonal_decomposition(y=trend, x=df['X'], period_range=[10, 20, 50, 100, 200, 500, 1000, 2000])

df_error = df.copy()
df_error["Y"] = cycle
df_error.to_csv('df_error.csv', index=False)

df_trend = df.copy()
df_trend["Y"] = result.trend
df_trend = df_trend.dropna()
df_trend.to_csv('df_trend.csv', index=False)

df_seasonal = df.copy()
df_seasonal["Y"] = result.seasonal
df_seasonal.to_csv('df_seasonal.csv', index=False)
