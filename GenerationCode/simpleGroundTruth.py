import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(0)

# Parameters
T = 100  # Number of time steps
mu_goal = np.array([10.0, 10.0])      # Mean of goal
cov_goal = np.eye(2)               # Covariance of goal

# Sample final goal from Gaussian
goal = np.random.multivariate_normal(mu_goal, cov_goal)

# Create linearly spaced trajectory from origin to goal
trajectory = np.linspace(start=np.array([3.0, 4.0]), stop=goal, num=T)

# Plot the trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Ground truth')
plt.scatter(*goal, color='red', label='Goal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Simulated 2D Trajectory')
plt.grid(True)
plt.show()

# Store as numpy array
np.save('../Data/ground_truth_trajectory.npy', trajectory)
