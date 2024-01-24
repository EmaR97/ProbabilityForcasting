import pandas as pd
from matplotlib import pyplot as plt

from Decompose import apply_hp_filter_with_optimal_lambda, seasonal_decomposition
from Fitting import (polynomial_fit_and_plot, seasonal_curve_fit_and_plot, seasonal_curve_fit_function,
                     complete_fit_and_plot, check_fitting_quality_and_print_metrics,
                     print_error_distribution_and_return_stats, )
from GaussianSurfaceProbability import evaluate_prob, create_surface, plot_surface

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

# Create a DataFrame for the seasonal component and save it to a CSV file
df_seasonal = df.copy()
df_seasonal["Y"] = result.seasonal

# Use polynomial_fit_and_plot to fit a polynomial to the trend component
poly_coef_ = polynomial_fit_and_plot(x=df_trend["X"].values.reshape(-1, 1), y=df_trend["Y"].values.reshape(-1, 1),
                                     degree=2)
# Use seasonal_curve_fit_and_plot to fit the function to the seasonal component
period_coef_ = seasonal_curve_fit_and_plot(f=seasonal_curve_fit_function, x=df_seasonal['X'], y=df_seasonal['Y'])

# Compose the fitted trend and seasonal component
y_fitted_, fitting_error_, fitted_function_ = complete_fit_and_plot(df["X"], df["Y"], poly_coef_, period_coef_)

# Check the quality of the fitting using the fitting error
check_fitting_quality_and_print_metrics(df["Y"], y_fitted_)

# Fit the error distribution with a gaussian
e_mean, e_std = print_error_distribution_and_return_stats(fitting_error_)

# Define integration limits for the double integral
x_lower, x_upper = 0, 10
y_lower, y_upper = -2, 2
y_lower_bound = 1

surface = create_surface(e_mean, e_std, fitted_function_)
# Calculate and print the probability of y > y_lower_bound within the specified x range
prob = evaluate_prob(x_lower, x_upper, y_lower, y_upper, surface, y_lower_bound)

# Create a 3D plot figure with two subplots
fig = plt.figure(figsize=(20, 20))

# Plot the surface for the specified range
plot_surface(fig, x_lower, x_upper, y_lower, y_upper, surface, 211, y_lower, y_upper, 0.1)

# Plot another surface for a different range in the second subplot
plot_surface(fig, x_lower, x_upper, y_lower_bound, y_upper, surface, 212, y_lower, y_upper, 0.1)

# Show the plot
plt.show()
