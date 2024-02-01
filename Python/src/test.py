from UtilityPlot import display_probability_surface
from gaussian_probability_estimation_new import calculate_mean_probability


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

    display_probability_surface(base_values, function, std_dev, threshold)


if __name__ == "__main__":
    main()
