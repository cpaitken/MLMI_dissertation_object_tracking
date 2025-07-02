import numpy as np
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
from dataFunctions import make_groundtruth, pretty_print_matrix, save_vector_arrays_txt, save_matrix_arrays_txt, save_state_comparison_txt, get_model_rmse, save_tracking_plot
from tqdm import tqdm

groundtruth = np.load('ground_truth_trajectory.npy') #This is fake for the moment, obviously
#groundtruth = make_groundtruth("Data/groundtruth.txt")
#groundtruth = groundtruth[::20]


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
G_prior = np.array([[10.0, 10.0]])
G_var = 1  #Unsure goal initially

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

#TAKE ANOTHER LOOK AT THIS - COULD BE WRONG INITIALIZATION
vk = [np.eye(d+1)]
vk[0][:-2, :-2] = f.iSE(t[1:], t[1:], 0.1, 1.0)
vk[0][-2, -2] = 0.1
vk[0][-1,-1] = G_var

#Storage for full debugging vectors
full_state_iSE = []
full_state_goal = []
all_F_aug = []
all_F_goal = []
all_m_predN = []
all_m_pred = []
all_m_updN = []
all_m_upd = []
#Update and covariance debugging
all_vpredN = []
all_v_updN = []
all_KGN = []
all_y_inN = []

all_vpred = []
all_v_upd = []
all_KG = []
all_y_in = []



XN = np.zeros([Tmax,2]) #Keep track of predicted state
#GN = np.zeros([Tmax, 2]) #Keep track of predicted goal
#Infered Goal Model Storage
X = np.zeros([Tmax,2]) #Keep track of predicted state
G = np.zeros([Tmax, 2]) #Keep track of predicted goal

#Inference Portion
#if non_goalModel:
for k in range(Tmax):
    m_predN, v_predN, F_aug = f.ise1_pred(t + dt * (k + 1), mkN[-1], vkN[-1], 0.1, 1.0)
    all_m_predN.append(m_predN.copy())
    all_vpredN.append(v_predN.copy())

    # One observation only
    y = noisy_data[k]

    # Association step (always associated here, for simplicity)
    datum = y

    # Update
    m_upN, v_upN, KG, y_in = f.update_ise1(datum, m_predN, v_predN, sy)
    all_m_updN.append(m_upN.copy())
    all_v_updN.append(v_upN.copy())
    all_KGN.append(KG.copy())
    all_y_inN.append(y_in.copy())


    # Record
    mkN.append(m_upN)
    vkN.append(v_upN)
    XN[k, :] = m_upN[0, :] + m_upN[-1, :]  # Estimate + inferred goal component
    #GN[k, :] = m_upN[-1, :]               # Just the inferred goal
    
    # Store full state vector for iSE model
    full_state_iSE.append(m_upN.copy())
    all_F_aug.append(F_aug.copy())
#else:
for k in range(Tmax):
    tk = t + k

    #Predict using the augmented state space model
    m_pred, v_pred, F_goal = iF.gise1_pred(t + dt * (k+1), mk[-1], vk[-1], 0.1, 1.0)
    all_m_pred.append(m_pred.copy())
    all_vpred.append(v_pred.copy())

    #Observation
    y = noisy_data[k]

    datum = y

    #Update with Kalman Filter
    m_up, v_up, KG, y_in = iF.augmented_update(datum, m_pred, v_pred, sy)
    all_m_upd.append(m_up.copy())
    all_v_upd.append(v_up.copy())
    all_KG.append(KG.copy())
    all_y_in.append(y_in.copy())



    mk.append(m_up)
    vk.append(v_up)
    X[k, :] = m_up[0, :] + m_up[-2, :] #Predicted most recent location
    #X[k, :] = m_up[0, :] + +m_up[-2, :] + m_up[-1, :] #Predicted most recent location and goal
    G[k, :] = m_up[-1, :] #Predicted goal
    
    # Store full state vector for goal model
    full_state_goal.append(m_up.copy())
    all_F_goal.append(F_goal.copy())

save_tracking_plot(groundtruth, noisy_data, X, G, XN, "plot_0.1betaTopRight_HwithoutGoal.png")

print(f"RMSE of iSE model:  {get_model_rmse(XN, groundtruth):.4f}")
print(f"RMSE of goal model:  {get_model_rmse(X, groundtruth):.4f}")


## Save detailed state information
# Convert lists to arrays for easier handling
full_state_iSE = np.array(full_state_iSE)  # Shape: (Tmax, d, 2)
full_state_goal = np.array(full_state_goal)  # Shape: (Tmax, d+1, 2)

# Save detailed state comparison
save_state_comparison_txt(full_state_iSE, full_state_goal, "detailed_state_comparison.txt")
save_matrix_arrays_txt(all_F_aug[:5], all_F_goal[:5], "transition_matrix_examples.txt", "F_aug", "F_goal")
save_vector_arrays_txt(all_m_predN, all_m_pred, "predicted_means.txt", "m_predN", "m_pred")
save_vector_arrays_txt(all_m_updN, all_m_upd, "updated_means.txt", "m_updN", "m_upd")
save_matrix_arrays_txt(all_vpredN[:5], all_vpred[:5], "prediction_noise_matrix_examples.txt", "V_predN", "VPred")
save_matrix_arrays_txt(all_v_updN[:5], all_v_upd[:5], "update_prediction_noise_matrix_examples.txt", "V_updN", "VUpdate")
save_vector_arrays_txt(all_KGN, all_KG, "kalman_gains.txt", "KGN", "KG")
save_vector_arrays_txt(all_y_inN, all_y_in, "innovations.txt", "y_inN", "y_in")

print("Saved debugging vectors and matrices to folder Debugging")



