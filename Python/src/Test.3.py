import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate synthetic data
X = np.linspace(start=0, stop=400, num=10000).reshape(-1, 1)
true_function = np.squeeze(0.01 * X + np.sin(X))
noise = np.random.normal(0, 0.1, size=X.shape[0])  # Adjust the standard deviation of the noise as needed
y = true_function + noise
# Plot settings
plt.figure(figsize=(20, 8), dpi=100)  # Adjust figure size and DPI for better quality
plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")

# Randomly select training data points
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=400, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

from statsmodels.tsa.filters.hp_filter import hpfilter

# Apply the HP filter
cycle, trend = hpfilter(y, lamb=2000)  # Adjust the smoothing parameter (lamb) as needed

# Plot the original time series, trend, and cycle
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(X, y, label='Original Data', color='blue')
plt.title('Original Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(X, trend, label='Trend', color='green')
plt.title('Trend Component')
plt.xlabel('Time')
plt.ylabel('Trend Value')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(X, cycle, label='Cycle', color='red')
plt.title('Cycle Component')
plt.xlabel('Time')
plt.ylabel('Cycle Value')
plt.legend()

plt.tight_layout()
plt.show()

# Apply seasonal decomposition
_result = seasonal_decompose(trend, period=1)  # Adjust the period as needed

# Plot the original time series, trend, seasonal, and residual components
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(X, y, label='Original Data', color='blue')
plt.title('Original Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(X, _result.trend, label='Trend', color='green')
plt.title('Trend Component')
plt.xlabel('Time')
plt.ylabel('Trend Value')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(X, _result.seasonal, label='Seasonal', color='red')
plt.title('Seasonal Component')
plt.xlabel('Time')
plt.ylabel('Seasonal Value')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(X, _result.resid, label='Residual', color='purple')
plt.title('Residual Component')
plt.xlabel('Time')
plt.ylabel('Residual Value')
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error

# Function to compute mean squared error for a given seasonal period
def mse_for_period(a, period):
    result = seasonal_decompose(a, period=period)
    reconstructed = result.trend + result.seasonal + result.resid
    return mean_squared_error(a, reconstructed)

# Try different period values and record the mean squared error
period_values = range(2, 101)  # Adjust the range based on your data characteristics
mse_values = []

for period in period_values:
    mse = mse_for_period(trend, period)
    mse_values.append(mse)

# Find the optimal period with the minimum mean squared error
optimal_period = period_values[np.argmin(mse_values)]

# Apply seasonal decomposition with the optimal period
_result = seasonal_decompose(trend, period=optimal_period)

# Plot the original time series, trend, seasonal, and residual components
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(X, trend, label='Original Data', color='blue')
plt.title('Original Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(X, _result.trend, label='Trend', color='green')
plt.title('Trend Component')
plt.xlabel('Time')
plt.ylabel('Trend Value')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(X, _result.seasonal, label='Seasonal', color='red')
plt.title(f'Seasonal Component (Period = {optimal_period})')
plt.xlabel('Time')
plt.ylabel('Seasonal Value')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(X, _result.resid, label='Residual', color='purple')
plt.title('Residual Component')
plt.xlabel('Time')
plt.ylabel('Residual Value')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Optimal Seasonal Period: {optimal_period}")