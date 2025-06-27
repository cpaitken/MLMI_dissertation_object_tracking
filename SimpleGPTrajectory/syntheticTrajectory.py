import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# === Parameters ===
n_points = 100
t = np.linspace(0, 10, n_points)
ell = 2.0             # Lengthscale for SE kernel
sigma = 1.0           # Output scale

#Starting Position
x0 = np.random.uniform(0,100)
start_point = np.array([x0, 0])

#Goal choosing
grid_size = 100
n_goals = 5
goals_centers = np.column_stack([
    np.random.uniform(0,100,n_goals),
    np.random.uniform(60,100,n_goals)
])

#2D Gaussian around the goals
goal_cov = np.diag([100,100])
chosen_goal = np.random.choice(n_goals)
G = np.random.multivariate_normal(goals_centers[chosen_goal], goal_cov)

# === SE Covariance Function ===
def squared_exp_kernel(t1, t2, ell, sigma):
    sqdist = cdist(t1[:, None], t2[:, None], 'sqeuclidean')
    return sigma**2 * np.exp(-0.5 * sqdist / ell**2)

K = squared_exp_kernel(t, t, ell, sigma)

# === Define mean and sample ===
mu_x = np.ones(n_points) * G[0]
mu_y = np.ones(n_points) * G[1]
sample_x = np.random.multivariate_normal(mu_x, K)
sample_y = np.random.multivariate_normal(mu_y, K)

# === Plot ===
plt.figure(figsize=(6, 6))
plt.plot(sample_x, sample_y, label="GP trajectory")
plt.scatter(G[0], G[1], color='black', marker='x', s=100, label='True goal (sampled)')
plt.scatter(sample_x[0], sample_y[0], color='red', label='Start')
plt.scatter(goals_centers[:,0], goals_centers[:,1], color='gray', label='Possible goal centers')
plt.title("2D GP Trajectory with Latent Goal from Mixture of Gaussians")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
