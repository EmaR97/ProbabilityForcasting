import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import seasonal_decompose

# Custom utility function for plotting time series data
from UtilityPlot import plot_time_series


# Function to apply the Hodrick-Prescott filter with optimal lambda
def apply_hp_filter_with_optimal_lambda(time_series, time, lambda_range):
    # Initialize variables to store the best results
    best_lambda = None
    best_trend = None
    best_cycle = None
    best_variance_ratio = float('inf')  # Initialize with a large value

    # Loop through different lambda values
    for lambda_ in lambda_range:
        # Apply Hodrick-Prescott filter
        cycle_, trend_ = hpfilter(time_series, lamb=lambda_)

        # Calculate variance ratio
        variance_ratio = np.var(cycle_) / np.var(time_series)

        # Check if the current result is better than the previous best
        if variance_ratio < best_variance_ratio:
            best_lambda = lambda_
            best_trend = trend_
            best_cycle = cycle_
            best_variance_ratio = variance_ratio

    # Plot the best result
    plt.figure(figsize=(10, 12))
    plt.subplot(3, 1, 1)
    plot_time_series(time, time_series, label='Original Time Series', title='Original Time Series')
    plt.subplot(3, 1, 2)
    plot_time_series(time, best_trend, label='Trend', title=f'Best Trend (lambda={best_lambda})', linestyle='--',
                     color='red')
    plt.subplot(3, 1, 3)
    plot_time_series(time, best_cycle, label='Cycle', title=f'Best Cycle (lambda={best_lambda})', linestyle='--',
                     color='green')
    plt.tight_layout()
    plt.show()

    return best_cycle, best_trend, best_lambda


# Function for seasonal decomposition
def seasonal_decomposition(time_series, time, period_range):
    # Initialize variables to store the best results
    best_period = None
    best_result = None
    best_seasonal_variance = float('-inf')  # Initialize with a small value
    best_residual_variance = float('inf')  # Initialize with a large value

    # Loop through different period values
    for period in period_range:
        # Apply seasonal decomposition
        result_ = seasonal_decompose(time_series, model='additive', period=period)

        # Calculate seasonal and residual variances
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
    plt.subplot(4, 1, 1)
    plot_time_series(time, time_series, label='Base Trend', title='Base Trend', linestyle='--', color='red')
    plt.subplot(4, 1, 2)
    plot_time_series(time, best_result.trend, label='Trend', title='Trend', linestyle='--', color='blue')
    plt.subplot(4, 1, 3)
    plot_time_series(time, best_result.seasonal, label='Seasonal', title='Seasonal', linestyle='--', color='blue')
    plt.subplot(4, 1, 4)
    plot_time_series(time, best_result.resid, label='Residual', title='Residual', linestyle='--', color='purple')
    plt.tight_layout()
    plt.show()

    return best_result


# Read data from a CSV file into a DataFrame
df = pd.read_csv('../../data/generated_data.csv')
time_, time_series_ = df["X"], df["Y"]

# Apply Hodrick-Prescott filter with optimal lambda
cycle, trend, optimal_lambda = apply_hp_filter_with_optimal_lambda(time_series=time_series_, time=time_,
                                                                   lambda_range=[1, 10, 50, 100, 200, 500, 0.5])
print(f"Optimal lambda: {optimal_lambda}")

# Perform seasonal decomposition on the trend component
result = seasonal_decomposition(time_series=trend, time=time_, period_range=[10, 20, 50, 100, 200, 500, 1000, 2000])

# Create a DataFrame for the trend component and save it to a CSV file
df_trend = df.copy()
df_trend["Y"] = result.trend
df_trend = df_trend.dropna()
df_trend.to_csv('../../data/df_trend.csv', index=False)

# Create a DataFrame for the seasonal component and save it to a CSV file
df_seasonal = df.copy()
df_seasonal["Y"] = result.seasonal
df_seasonal.to_csv('../../data/df_seasonal.csv', index=False)
