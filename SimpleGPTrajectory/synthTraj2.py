import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# === Parameters ===
n_points = 300
t = np.linspace(0, 10, n_points + 1)  # include t0
ell = 2.0
sigma = 1.0
u = np.array([80.0, 90.0])  # GP mean (goal)
x_start = np.array([10.0, 10.0])  # fixed start location

# === SE Kernel ===
def se_kernel(t1, t2, ell, sigma):
    sqdist = cdist(t1[:, None], t2[:, None], 'sqeuclidean')
    return sigma**2 * np.exp(-0.5 * sqdist / ell**2)

K_full = se_kernel(t, t, ell, sigma)
mu_full = np.ones(len(t))  # mean 1-vector for the GP

# === Conditioning: indices ===
K_00 = K_full[0:1, 0:1].reshape(1, 1)
K_0T = K_full[0:1, 1:]      # shape (1, T)
K_T0 = K_full[1:, 0:1]       
K_TT = K_full[1:, 1:]     # shape (T, T)

# === Conditioned GP for x and y ===
def sample_conditioned_GP(start_val, mean_val):
    mu_0 = mean_val
    mu_T = np.ones(n_points) * mean_val

    cond_mean = mu_T + (K_T0 @ np.linalg.inv(K_00))[:,0] * (start_val - mu_0)
    cur_inside = K_T0 @ np.linalg.inv(K_00)
    cond_cov = K_TT - (K_T0 @ np.linalg.inv(K_00)) @ K_0T

    return np.random.multivariate_normal(cond_mean.ravel(), cond_cov)

sample_x = np.concatenate([[x_start[0]], sample_conditioned_GP(x_start[0], u[0])])
sample_y = np.concatenate([[x_start[1]], sample_conditioned_GP(x_start[1], u[1])])

# === Plot ===
plt.figure(figsize=(6,6))
plt.plot(sample_x, sample_y, label='GP trajectory')
plt.scatter(*u, color='black', marker='x', s=100, label='Goal (mean)')
plt.scatter(*x_start, color='red', label='Start (fixed)')
plt.title("2D GP Trajectory with Fixed Start and Mean Goal")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
