import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # Assuming seaborn is installed
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import seasonal_decompose

# Create a DataFrame
df = pd.read_csv('generated_data.csv')
X, Y = df["X"], df["Y"]


def plot_time_series(x, y, label=None, xlabel='X', ylabel='Values', title=None, linestyle='-', color='blue'):
    plt.plot(x, y, label=label, linestyle=linestyle, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()


def apply_hp_filter_with_optimal_lamb(y, x, lamb_range):
    best_lamb = None
    best_trend = None
    best_cycle = None
    best_variance_ratio = float('inf')  # Initialize with a large value

    for lamb in lamb_range:
        cycle, trend = hpfilter(y, lamb=lamb)
        variance_ratio = np.var(cycle) / np.var(y)

        # Check if the current result is better than the previous best
        if variance_ratio < best_variance_ratio:
            best_lamb = lamb
            best_trend = trend
            best_cycle = cycle
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


lamb_range = [1, 10, 50, 100, 200, 500, 0.5]  # Adjust the range as needed
cycle, trend, best_lamb = apply_hp_filter_with_optimal_lamb(y=df['Y'], x=df['X'], lamb_range=lamb_range)
print(f"Optimal lamb: {best_lamb}")


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


result = seasonal_decomposition(y=trend, x=df['X'], period_range=[10, 20, 50, 100, 200, 500, 1000, 2000])


# Curve fitting using scipy.optimize.curve_fit
def func(x, a, b, c):
    return a * x + b * np.sin(c * x)


def curve_fit_function(f, x, y):
    res = curve_fit(f, x, y)
    # Extract the fitted parameters
    a_fit, b_fit, c_fit = res[0]
    # Create a new column in the DataFrame with the fitted values
    df['y_fit'] = f(x, a_fit, b_fit, c_fit)
    print("Curve fit coefficients :", res[0])

    # Plot the original seasonal data and the fitted curve
    # Predicted data plot
    plt.figure(figsize=(20, 20))
    # Plot Base Trend
    plt.subplot(2, 1, 1)
    plot_time_series(x, y, label='Base', title='Base', linestyle='-', color='red')
    # Plot Trend
    plt.subplot(2, 1, 2)
    plot_time_series(x, df['y_fit'], label='Fitted', title='Curve Fit', linestyle='--', color='blue')
    plt.show()

    return a_fit, b_fit, c_fit


# Use curve_fit to fit the function to the seasonal component
c_c = curve_fit_function(f=func, x=df['X'], y=result.seasonal)

df2 = df.copy()
df2["seasonal_trend"] = result.trend
df2 = df2.dropna()


def polynomial_fit(x, y, degree=1):
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


p_c = polynomial_fit(x=df2["X"].values.reshape(-1, 1), y=df2["seasonal_trend"].values.reshape(-1, 1), degree=2)


def fit_complete(poly_coef=p_c, period_coef=c_c):
    f = np.squeeze(
        poly_coef[0] + poly_coef[1] * X + poly_coef[2] * X ** 2 + period_coef[0] * X + period_coef[1] * np.sin(
            period_coef[2] * X))
    plt.figure(figsize=(20, 20))
    # Plotting the base trend
    plt.subplot(3, 1, 1)
    plot_time_series(X, Y, title='Base', linestyle='-', color='red')
    # Plotting the fitted trend
    plt.subplot(3, 1, 2)
    plot_time_series(X, f, title='Fitted', linestyle='--', color='blue')
    # Plotting the error (difference) between base and fitted trends
    plt.subplot(3, 1, 3)
    e = Y - f
    plot_time_series(X, e, title='Error', linestyle='-', color='green')
    plt.show()
    return f, e


fitted_function, error = fit_complete()


def check_fitting_quality():
    mse = mean_squared_error(Y, fitted_function)
    rmse = np.sqrt(mse)
    r_squared = r2_score(Y, fitted_function)
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r_squared}')


check_fitting_quality()


# Assuming 'error' is the array of differences between the base and fitted trends
def print_error_distribution(e):
    # Plotting a histogram of errors
    plt.figure(figsize=(20, 20))
    sns.histplot(e, bins=30, kde=True, color='green')
    plt.title('Distribution of Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.show()
    e_mean, e_std = norm.fit(e)
    print(f"Mean of the error: {e_mean}")
    print(f"Standard deviation of the error: {e_std}")
    return e_mean, e_std


error_mean, error_std = print_error_distribution(error)
