import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def calculate_probability(mean, std_dev, threshold):
    """
    Calculate the probability that a random variable exceeds a given threshold.

    Parameters:
    - mean: Mean of the random variable.
    - std_dev: Standard deviation of the random variable.
    - threshold: Threshold value.

    Returns:
    - The probability that the random variable is greater than the threshold, in percentage.
    """
    # Calculate the z-score
    z_score = (threshold - mean) / std_dev
    # Calculate the probability using the complementary cumulative distribution function (CDF)
    probability = 1 - norm.cdf(z_score)
    # Convert the probability to a percentage and round it
    return probability * 100


def evaluate_points(function, lower_bound, upper_bound, num_points=100):
    """
    Generate the mean values within the specified range using the given function and number of points.

    Parameters:
    - function: The function to evaluate.
    - lower_bound: Lower bound of the range.
    - upper_bound: Upper bound of the range.
    - num_points: Number of points to generate within the range.

    Returns:
    - mean_values: List of mean values evaluated from the function.
    - base_values: List of base values (independent variable) within the range.
    """
    base_values = np.linspace(lower_bound, upper_bound, num_points)
    return [function(x) for x in base_values], base_values


def calculate_mean_probability(function, std_dev, lower_bound, upper_bound, threshold, num_points=100):
    """
    Calculate the mean probability that a function of a random variable exceeds a threshold within a given range.

    Parameters:
    - function: The function representing the relationship between the random variable and another variable.
    - std_dev: Standard deviation of the random variable.
    - lower_bound: Lower bound of the range for the independent variable.
    - upper_bound: Upper bound of the range for the independent variable.
    - threshold: Threshold value.
    - num_points: Number of points used for approximation within the range.

    Returns:
    - mean_probability: The mean probability that the function of the random variable exceeds the threshold within
    the given range.
    - probabilities: List of probabilities corresponding to each mean value within the range.
    - mean_values: List of mean values within the range [lower_bound, upper_bound].
    - base_values: List of base values (independent variable) within the range.
    """
    # Generate the mean values within the range [lower_bound, upper_bound]
    mean_values, base_values = evaluate_points(function, lower_bound, upper_bound, num_points)
    # Calculate probabilities for each x in the range [lower_bound, upper_bound]
    probabilities = [calculate_probability(x, std_dev, threshold) for x in mean_values]
    # Calculate the mean probability
    mean_probability = round(sum(probabilities) / len(probabilities), 2)
    return mean_probability, probabilities, mean_values, base_values


def display_probability_surface(base_values, function, std_dev, threshold):
    """
    Display the probability surface and function curve.

    Args:
    - base_values: array-like, the base values of the independent variable
    - function: function, the function to visualize
    - std_dev: float, standard deviation for the normal distribution
    - threshold: float, threshold value for the probability distribution
    """

    # Independent variable
    x = base_values

    # Determine plot ranges
    y_lower_limit, y_upper_limit = min(min(function(x)) - 2 * std_dev, threshold - std_dev), max(
        max(function(x)) + 2 * std_dev, threshold + std_dev)

    # Values for dependent variable
    y = np.linspace(y_lower_limit, y_upper_limit, 100)

    # Create the figure
    fig = plt.figure(figsize=(20, 10))

    # Subplot 1 for the function curve
    ax1 = fig.add_subplot(121)
    ax1.plot(x, function(x), label='Function Curve')
    ax1.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    ax1.set_xlabel('Independent Variable (x)')
    ax1.set_ylabel('Dependent Variable (y)')
    ax1.set_title('Function Curve')
    ax1.set_ylim([y_lower_limit, y_upper_limit])
    ax1.grid(True)
    ax1.legend()

    # Create a meshgrid for 3D plotting
    X, Y = np.meshgrid(x, y)

    # Compute the normal probability density function
    Z1 = norm.pdf(Y, loc=function(X), scale=std_dev)

    # Create a copy for areas where y <= threshold
    Z2 = np.copy(Z1)
    Z2[Y <= threshold] = np.nan

    # Subplot 2 for the comparison of surfaces
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(elev=80, azim=-90)
    ax2.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.7, label='Original Surface')
    ax2.plot_surface(X, Y, Z2, color='red', alpha=1, label='Y > Threshold Surface')
    ax2.set_xlabel('Independent Variable (x)')
    ax2.set_ylabel('Dependent Variable (y)')
    ax2.set_zlabel('PDF')
    ax2.set_title('Comparison of Surfaces')
    ax2.set_ylim([y_lower_limit, y_upper_limit])
    ax2.legend()

    # Show the plot
    plt.show()


def main():
    # Parameters
    std_dev = 1  # Standard deviation of the error distribution
    threshold = 4  # Threshold for y
    lower_bound = 5
    upper_bound = 15

    # Define the function relating x and y
    def function(x):
        return x ** 0.5  # Example function, replace with your own

    # Calculate the mean probability and get the curve values
    mean_probability, _, _, base_values = calculate_mean_probability(function, std_dev, lower_bound, upper_bound,
                                                                     threshold, 1000)
    print(f"Mean probability of y > {threshold} given {lower_bound} < x < {upper_bound}: {mean_probability}%")

    display_probability_surface(base_values, function, std_dev, threshold)


if __name__ == "__main__":
    main()
