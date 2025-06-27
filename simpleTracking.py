import numpy as np
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
from dataFunctions import make_groundtruth
from tqdm import tqdm

#groundtruth = np.load('ground_truth_trajectory.npy') #This is fake for the moment, obviously
groundtruth = make_groundtruth("Data/groundtruth.txt")
groundtruth = groundtruth[::20]


## Hyperparameter specification ##
d = 5 #Sliding window
s2 = 1.0 #Prior output variance
ls = 3 #Length scale
T = groundtruth.shape[0]
Tmax = T #Maximum steps
dt = 1
t = dt * np.arange(d, 0, -1) #Time vector with most recent measurement first
assoc_threshold = 5

##Adding noise to create simulated sensor measurements if only groundtruth data available
sy = 0.5
noisy_data = [groundtruth[k] + np.random.normal(0, sy, 2) for k in range(Tmax)]


## Goal Creation ##
G_prior = np.array([[10.0, 1.0]])
G_var = 5  #Unsure goal initially

#if non_goalModel:
mkN = [groundtruth[0, :] * np.ones([d, 2])]
mkN[0][:-1] -= mkN[0][-1]  # iSE-1 style offset
vkN = [np.eye(d)]
vkN[0][:-1, :-1] = f.iSE(t[1:], t[1:], 0.1, 1.0)
vkN[0][-1, -1] = 0.1
#print("Shape of mkN:", mkN[0].shape)
#else:
    ## Storage Variables ##
mk = [groundtruth[0, :] * np.ones([d, 2])]
mk[0][:-1] -= mk[0][-1]
mk[0] = np.vstack((mk[0], G_prior))
#print("Shape of mk:", mk[0].shape)
vk = [np.eye(d+1)]
vk[0][:-2, :-2] = f.iSE(t[1:], t[1:], 0.1, 1.0)
vk[0][-2, -2] = 0.1
vk[0][-1,-1] = G_var

XN = np.zeros([Tmax,2]) #Keep track of predicted state
GN = np.zeros([Tmax, 2]) #Keep track of predicted goal
#Infered Goal Model Storage
X = np.zeros([Tmax,2]) #Keep track of predicted state
G = np.zeros([Tmax, 2]) #Keep track of predicted goal

#Inference Portion
#if non_goalModel:
for k in range(Tmax):
    m_predN, v_predN = f.ise1_pred(t + dt * (k + 1), mkN[-1], vkN[-1], 0.1, 1.0)
    
    # One observation only
    y = noisy_data[k]

    # Association step (always associated here, for simplicity)
    datum = y

    # Update
    m_upN, v_upN = f.update_ise1(datum, m_predN, v_predN, sy)

    # Record
    mkN.append(m_upN)
    vkN.append(v_upN)
    XN[k, :] = m_upN[0, :] + m_upN[-1, :]  # Estimate + inferred goal component
    GN[k, :] = m_upN[-1, :]               # Just the inferred goal
#else:
for k in range(Tmax):
    tk = t + k

    #Predict using the augmented state space model
    m_pred, v_pred = iF.gise1_pred(t + dt * (k+1), mk[-1], vk[-1], 0.1, 1.0)

    #Observation
    y = noisy_data[k]

    datum = y

    #Update with Kalman Filter
    m_up, v_up, = iF.augmented_update(datum, m_pred, v_pred, sy)

    mk.append(m_up)
    vk.append(v_up)
    X[k, :] = m_up[0, :] + m_up[-1, :] #Predicted most recent location
    G[k, :] = m_up[-1, :] #Predicted goal

plt.plot(groundtruth[:,0], groundtruth[:,1], label='Truth')
plt.scatter(*zip(*noisy_data), alpha=0.3, label='Noisy obs')
#plt.plot(noisy_data[:,0], noisy_data[:,1], 'o', label='Noisy obs', alpha=0.3)
plt.plot(X[:,0], X[:,1], label='Estimate Goal Model', color='green')
plt.plot(G[:,0], G[:,1], label='Inferred goal', color='darkred')
plt.plot(XN[:,0], XN[:,1], '--', label='iSE Estimate Goal Model', color='limegreen')
plt.plot(GN[:,0], GN[:,1], '--', label='iSE Goal?', color='firebrick')
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
