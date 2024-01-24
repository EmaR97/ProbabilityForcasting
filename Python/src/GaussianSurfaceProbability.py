# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


# Define a Gaussian function with parameters amp_, mean, and std_
def gaussian(x, y, amp_, mean, std_):
    return amp_ * np.exp(-(((y - mean(x)) ** 2) / (2 * (std_ ** 2))))


# Define a surface function using the Gaussian function
def surface(y, x):
    return gaussian(x, y, amp_=amp, std_=std, mean=trend)


# Calculate the probability of y > y_lower_bound within the specified x range
def evaluate_prob(x_l, x_u, y_l, y_u, surf, y_l_b):
    # Perform double integration to calculate segment volume and total volume
    vol_seg = integrate.dblquad(surf, x_l, x_u, y_l_b, y_u)
    print(f"Segment volume and error: {vol_seg}")
    vol_tot = integrate.dblquad(surf, x_l, x_u, y_l, y_u)
    print(f"Total volume and error: {vol_tot}")
    # Calculate the probability of y > y_lower_bound within the specified x range
    seg_frac = (vol_seg[0] / vol_tot[0]) * 100
    res = round(seg_frac) if seg_frac >= 1 else "less than 1"
    print(f"Probability of y>{y_l_b} for {x_l}<x<{x_u} estimated to {res}%")
    return res


# Define a function to plot a 3D surface plot
def plot_surface(x_l, x_u, y_l, y_u, surf, position):
    ax = fig.add_subplot(position, projection='3d')

    # Create a set of data points for x and y
    x = np.linspace(x_l, x_u, 1000)
    y = np.linspace(y_l, y_u, 1000)
    x, y = np.meshgrid(x, y)

    # Compute the values of the surface function
    z = surf(y=y, x=x)

    # Plot the 3D surface with customizations
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.view_init(elev=0, azim=0)  # Set the viewing angle
    ax.set_ylim([-20, 20])
    ax.set_zlim([0, 1])

    # Add labels to the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


# Set parameters for the Gaussian function and the surface plot
amp, std, trend = 1, 2, lambda x: 0.1 * x + 2 + 3 * np.sin(2 * x) + 0.5 * np.sin(20 * x)

# Define integration limits for the double integral
x_lower, x_upper = 0, 10
y_lower, y_upper = -20, 20
y_lower_bound = 5

# Calculate and print the probability of y > y_lower_bound within the specified x range
prob = evaluate_prob(x_lower, x_upper, y_lower, y_upper, surface, y_lower_bound)

# Create a 3D plot figure with two subplots
fig = plt.figure(figsize=(20, 20))

# Plot the surface for the specified range
plot_surface(x_lower, x_upper, y_lower, y_upper, surface, 211)

# Plot another surface for a different range in the second subplot
plot_surface(x_lower, x_upper, y_lower_bound, 20, surface, 212)

# Show the plot
plt.show()
