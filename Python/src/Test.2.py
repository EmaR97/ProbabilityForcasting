import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data for demonstration
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + np.random.randn(100) * 1.5

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Gaussian Process model
kernel = 1.0 * RBF() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

# Train the model
gpr.fit(X_train, y_train)

# Make predictions and get uncertainty estimates
y_pred_mean, y_pred_std = gpr.predict(X_test, return_std=True)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_mean)
print(f'Mean Squared Error: {mse}')

# Plot the predictions with uncertainty
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred_mean, color='red', label='Predicted Mean')
plt.fill_between(X_test.squeeze(), y_pred_mean - 1.96 * y_pred_std, y_pred_mean + 1.96 * y_pred_std, color='lightblue', alpha=0.2, label='95% Confidence Interval')
plt.legend()
plt.show()
