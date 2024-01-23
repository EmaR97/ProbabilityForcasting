import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate a sample time series data (replace this with your actual data)
np.random.seed(42)
time_index = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
values = np.random.randn(len(time_index))
df = pd.DataFrame({'Date': time_index, 'Value': values})

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Create a Gaussian Process regression model with RBF kernel
kernel = 1.0 * RBF(length_scale=1.0)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

# Train the model
model.fit(train.index.values.reshape(-1, 1), train['Value'].values)

# Make predictions and obtain uncertainty estimates
forecast_mean, forecast_std = model.predict(test.index.values.reshape(-1, 1), return_std=True)

# Calculate mean squared error on the test set
mse = mean_squared_error(test['Value'], forecast_mean)
print(f"Mean Squared Error on Test Set: {mse}")

# Plot the actual vs. predicted values with uncertainty
plt.plot(train['Date'], train['Value'], label='Training Data')
plt.plot(test['Date'], test['Value'], label='Test Data')
plt.plot(test['Date'], forecast_mean, label='Forecast', linestyle='dashed')
plt.fill_between(test['Date'], forecast_mean - forecast_std, forecast_mean + forecast_std, alpha=0.2, color='gray')
plt.legend()
plt.show()
