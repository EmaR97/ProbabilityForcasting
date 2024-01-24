import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from UtilityPlot import plot_time_series


# Function for curve fitting using scipy.optimize.curve_fit
def seasonal_curve_fit_function(x, a, b, c):
    return a * x + b * np.sin(c * x)


def seasonal_curve_fit_and_plot(f, x, y):
    res = curve_fit(f, x, y)
    # Extract the fitted parameters
    a, b, c = res[0]
    # Create a new column in the DataFrame with the fitted values
    y_fit = f(x, a, b, c)
    print("Curve fit coefficients :", res[0])
    # Plot the original seasonal data and the fitted curve
    # Predicted data plot
    plt.figure(figsize=(20, 20))
    # Plot Base Trend
    plt.subplot(2, 1, 1)
    plot_time_series(x, y, label='Base', title='Base', linestyle='-', color='red')
    # Plot Trend
    plt.subplot(2, 1, 2)
    plot_time_series(x, y_fit, label='Fitted', title='Curve Fit', linestyle='--', color='blue')
    plt.show()

    return a, b, c


# Function for polynomial fitting
def polynomial_fit_and_plot(x, y, degree=1):
    pr = PolynomialFeatures(degree=degree)
    x_poly = pr.fit_transform(x)
    lr_2 = LinearRegression()
    lr_2.fit(x_poly, y)

    # Predicted data plot
    plt.figure(figsize=(20, 20))
    # Plot Base Trend
    plt.subplot(2, 1, 1)
    plot_time_series(x, y, title='Base', linestyle='-', color='red')
    # Plot Trend
    plt.subplot(2, 1, 2)
    plot_time_series(x, lr_2.predict(x_poly), title='Polynomial Regression', linestyle='--', color='blue')
    plt.show()
    # Get the coefficients
    coefficients = lr_2.coef_[0]
    print("Polynomial coefficients :", coefficients)
    return coefficients


# Function for complete fitting
def complete_fit_and_plot(x, y, poly_coef, period_coef):
    fitted_function = lambda x: poly_coef[0] + poly_coef[1] * x + poly_coef[2] * x ** 2 + period_coef[0] * x + \
                                period_coef[1] * np.sin(period_coef[2] * x)
    y_fitted = np.squeeze(fitted_function(y))
    plt.figure(figsize=(20, 20))
    # Plotting the base trend
    plt.subplot(3, 1, 1)
    plot_time_series(x, y, title='Base', linestyle='-', color='red')
    # Plotting the fitted trend
    plt.subplot(3, 1, 2)
    plot_time_series(x, y_fitted, title='Fitted', linestyle='--', color='blue')
    # Plotting the error (difference) between base and fitted trends
    plt.subplot(3, 1, 3)
    fitting_error = y - y_fitted
    plot_time_series(x, fitting_error, title='Error', linestyle='-', color='green')
    plt.show()
    return y_fitted, fitting_error, fitted_function


# Function to check fitting quality
def check_fitting_quality_and_print_metrics(y, y_fitted):
    mse = mean_squared_error(y, y_fitted)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y, y_fitted)
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r_squared}')


# Function to print error distribution
def print_error_distribution_and_return_stats(fitting_error):
    plt.figure(figsize=(20, 20))
    sns.histplot(fitting_error, bins=30, kde=True, color='green')
    plt.title('Distribution of Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.show()

    error_mean, error_std = norm.fit(fitting_error)
    print(f"Mean of the error: {error_mean}")
    print(f"Standard deviation of the error: {error_std}")
    return error_mean, error_std


df = pd.read_csv('../../data/generated_data.csv')

df_trend = pd.read_csv('../../data/df_trend.csv')

df_seasonal = pd.read_csv('../../data/df_seasonal.csv')

poly_coef_ = polynomial_fit_and_plot(x=df_trend["X"].values.reshape(-1, 1), y=df_trend["Y"].values.reshape(-1, 1),
                                     degree=2)
# Use curve_fit to fit the function to the seasonal component
period_coef_ = seasonal_curve_fit_and_plot(f=seasonal_curve_fit_function, x=df_seasonal['X'], y=df_seasonal['Y'])

y_fitted_, fitting_error_, fitted_function_ = complete_fit_and_plot(df["X"], df["Y"], poly_coef_, period_coef_)

check_fitting_quality_and_print_metrics(df["Y"], y_fitted_)

e_mean, e_std = print_error_distribution_and_return_stats(fitting_error_)
