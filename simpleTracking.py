import numpy as np
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
from tqdm import tqdm

groundtruth = np.load('ground_truth_trajectory.npy') #This is fake for the moment, obviously

##Adding noise to create simulated sensor measurements if only groundtruth data available
sy = 0.5
noisy_data = groundtruth + np.random.randn(*groundtruth.shape) * sy**0.5

## Hyperparameter specification ##
d = 10 #Sliding window
s2 = 1.0 #Prior output variance
ls = 3 #Length scale
T = groundtruth.shape[0]
t = np.arange(d, 0, -1) #Time vector with most recent measurement first


## Goal Creation ##
G_prior = np.array([5.0, 5.0])
G_var = 10  #Unsure goal initially

# Initial mean with augmented state
m0 = np.zeros((d+1, 2))  #Length of sliding window, plus extra for goal, in 2D
m0[:d, :] = np.tile(groundtruth[0], (d,1)) #Repeating the first position for track initialization
m0[-1, :] = G_prior

## Initialize covariance ##
v0 = np.zeros((d+1, d+1))
v0[:d, :d] = f.iSE(t,t,s2,ls)
v0[-1, -1] = G_var

## Storage Variables ##
mk = [m0]
vk = [v0]

X = np.zeros((T,2)) #Keep track of predicted state
G = np.zeros((T, 2)) #Keep track of predicted goal

#Inference Portion
for k in range(T):
    tk = t + k

    #Predict using the augmented state space model
    m_pred, v_pred = iF.gise1_pred(tk, mk[-1], vk[-1], s2, ls)

    #Observation
    y = noisy_data[k]

    #Update with Kalman Filter
    m_up, v_up, = iF.augmented_update(y, m_pred, v_pred, sy)

    mk.append(m_up)
    vk.append(v_up)
    X[k] = m_up[0] #Predicted most recent location
    G[k] = m_up[-1] #Predicted goal

plt.plot(groundtruth[:,0], groundtruth[:,1], label='Truth')
plt.plot(noisy_data[:,0], noisy_data[:,1], 'o', label='Noisy obs', alpha=0.3)
plt.plot(X[:,0], X[:,1], label='Estimate')
plt.plot(G[:,0], G[:,1], label='Inferred goal')
plt.legend()
plt.show()

## Saving the states and goals for troubleshooting
save_path = "tracking_results.txt"

with open(save_path, 'w') as f:
    f.write("Estimated states\n")
    for x in X:
        f.write(f"{x[0]:.2f}, {x[1]:.2f}\n")
    f.write("\nInferred Goals\s")
    for g in G:
        f.write(f"{g[0]:.2f}, {g[1]:.2f}\n")
