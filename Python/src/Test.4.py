import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate synthetic data
X = np.linspace(start=0, stop=200, num=2000).reshape(-1, 1)
true_function = np.squeeze(0.01 * X + np.sin(X) + 0.5 * np.sin(5 * X))
noise = np.random.normal(0, 0.001, size=X.shape[0])
Y = true_function + noise

# Create a DataFrame
df = pd.DataFrame({'X': np.squeeze(X), 'Y': Y})

# Display the DataFrame
print(df.head())

# Apply the HP filter
cycle, trend = hpfilter(df['Y'], lamb=2000)  # Adjust the smoothing parameter (lamb) as needed

# Plot the original time series
plt.figure(figsize=(10, 12))

# Plot Original Time Series
plt.subplot(3, 1, 1)
plt.plot(df['X'], df['Y'], label='Original Time Series')
plt.xlabel('X')
plt.ylabel('Values')
plt.title('Original Time Series')

# Plot Trend
plt.subplot(3, 1, 2)
plt.plot(df['X'], trend, label='Trend', linestyle='--', color='red')
plt.xlabel('X')
plt.ylabel('Values')
plt.title('Trend')

# Plot Cycle
plt.subplot(3, 1, 3)
plt.plot(df['X'], cycle, label='Cycle', linestyle='--', color='green')
plt.xlabel('X')
plt.ylabel('Values')
plt.title('Cycle')

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# Seasonal decomposition
_result = seasonal_decompose(trend, model='additive', period=1000)  # Adjust the period as needed

# Plot the original time series
plt.figure(figsize=(20, 20))

# Plot Trend
plt.subplot(4, 1, 1)
plt.plot(df['X'], trend, label='Trend', linestyle='--', color='red')
plt.xlabel('X')
plt.ylabel('Values')
plt.title('Base')

# Plot Seasonal
plt.subplot(4, 1, 2)
plt.plot(df['X'], _result.trend, label='Trend', linestyle='--', color='blue')
plt.xlabel('X')
plt.ylabel('Values')
plt.title('Trend')

# Plot Seasonal
plt.subplot(4, 1, 3)
plt.plot(df['X'], _result.seasonal, label='Seasonal', linestyle='--', color='blue')
plt.xlabel('X')
plt.ylabel('Values')
plt.title('Seasonal')

# Plot Residual
plt.subplot(4, 1, 4)
plt.plot(df['X'], _result.resid, label='Residual', linestyle='--', color='purple')
plt.xlabel('X')
plt.ylabel('Values')
plt.title('Residual')

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

from scipy.optimize import curve_fit


# Define the function to fit
def func(x, a, b, c):
    return a * x + b * np.sin(c * x)


# Use curve_fit to fit the function to the data
res = curve_fit(func, df['X'], _result.seasonal)

# Extract the fitted parameters
a_fit, b_fit, c_fit = res[0]

# Create a new column in the DataFrame with the fitted values
df['y_fit'] = func(df['X'], a_fit, b_fit, c_fit)

# Plot the original data and the fitted curve
plt.scatter(df['X'], _result.seasonal, label='Original Data')
plt.plot(df['X'], df['y_fit'], label='Fitted Curve', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
