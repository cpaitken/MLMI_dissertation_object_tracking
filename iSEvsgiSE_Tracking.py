import numpy as np
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
from dataFunctions import make_groundtruth, pretty_print_matrix, save_vector_arrays_txt, save_matrix_arrays_txt, save_state_comparison_txt, get_model_rmse, save_tracking_plot, save_variance_array_txt
from tqdm import tqdm


groundtruth_filename = "Data/Generated/SE_track.txt"
debugging_folder = "Debugging/iSE/Generated/goal_iSE_track"
UZH = False
#groundtruth = groundtruth[::20]
groundtruth = make_groundtruth(groundtruth_filename, UZH=UZH)
if UZH:
    groundtruth = groundtruth[::20]

## Hyperparameter specification ##
d = 5 #Sliding window
s2 = 100 #Prior output variance
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
vkN[0][:-1, :-1] = f.iSE(t[1:], t[1:], s2/10, ls)
vkN[0][-1, -1] = s2/10
#print("Shape of mkN:", mkN[0].shape)
#else:
    ## Storage Variables ##
mk = [groundtruth[0, :] * np.ones([d, 2])]
mk[0][:-1] -= mk[0][-1]
mk[0] = np.vstack((mk[0], G_prior))

#TAKE ANOTHER LOOK AT THIS - COULD BE WRONG INITIALIZATION
vk = [np.eye(d+1)]
vk[0][:-2, :-2] = f.iSE(t[1:], t[1:], s2/10, ls)
vk[0][-2, -2] = s2/10
vk[0][-1,-1] = G_var

#Storage for full debugging vectors
normal_F_aug = []
goal_F_aug = []

normal_predicted_means = []
goal_predicted_means = []
normal_updated_means = []
goal_updated_means = []



X_normal = np.zeros([Tmax,2]) #Keep track of predicted state
S_normal = np.zeros([Tmax])
#Infered Goal Model Storage
X_goal = np.zeros([Tmax,2]) #Keep track of predicted state
S_goal = np.zeros([Tmax])
G_goal = np.zeros([Tmax, 2]) #Keep track of predicted goal
S_goal_var = np.zeros([Tmax])

#Inference Portion
#if non_goalModel:
for k in range(Tmax):
    m_predN, v_predN, F_aug = f.ise1_pred(t, mkN[-1], vkN[-1], 0.1, 1.0)
    

    # One observation only
    y = noisy_data[k]

    # Association step (always associated here, for simplicity)
    datum = y

    # Update
    m_upN, v_upN, KG, y_in = f.update_ise1(datum, m_predN, v_predN, sy)

    # Record
    mkN.append(m_upN)
    vkN.append(v_upN)

    X_normal[k, :] = m_upN[0, :] + m_upN[-1, :]  # Estimate + inferred goal component
    S_normal[k] = v_upN[0,0] + v_upN[-1,-1]
    
    # Store full state vector for iSE model
    normal_F_aug.append(F_aug.copy())
    normal_predicted_means.append(m_predN.copy())
    normal_updated_means.append(m_upN.copy())
#else:
for k in range(Tmax):
    #Predict using the augmented state space model
    m_pred, v_pred, F_goal = iF.gise1_pred(t, mk[-1], vk[-1], 0.1, 1.0)

    #Observation
    y = noisy_data[k]
    datum = y

    #Update with Kalman Filter
    m_up, v_up, KG, y_in = iF.augmented_update(datum, m_pred, v_pred, sy)

    mk.append(m_up)
    vk.append(v_up)
    X_goal[k, :] = m_up[0, :] + m_up[-2, :] + m_up[-1, :] #Predicted most recent location
    S_goal[k] = v_up[0,0] + v_up[-2,-2] + v_up[-1,-1]
    S_goal_var[k] = v_up[-1,-1]#X[k, :] = m_up[0, :] + +m_up[-2, :] + m_up[-1, :] #Predicted most recent location and goal
    G_goal[k, :] = m_up[-1, :] #Predicted goal
    
    # Store full state vector for goal model
    goal_F_aug.append(F_goal.copy())
    goal_predicted_means.append(m_pred.copy())
    goal_updated_means.append(m_up.copy())

save_tracking_plot(groundtruth, noisy_data, X_goal, G_goal, X_normal, "Goal-iSE", "iSE", "ComparisonPlot.png", debugging_folder)
save_matrix_arrays_txt(normal_F_aug[:5], goal_F_aug[:5], "transitionMatrices.txt", "F_aug", "F_goal", debugging_folder)
save_vector_arrays_txt(normal_predicted_means, goal_predicted_means, "predictedMeans.txt", "m_predN", "m_pred", debugging_folder)
save_vector_arrays_txt(normal_updated_means, goal_updated_means, "updatedMeans.txt", "m_updN", "m_upd", debugging_folder)
save_variance_array_txt(S_goal_var, "goalVariances.txt", debugging_folder)
save_variance_array_txt(S_normal, "normalLocationVariances.txt", debugging_folder)
save_variance_array_txt(S_goal, "goalLocationVariances.txt", debugging_folder)

se_rmse = get_model_rmse(X_normal, groundtruth)
gse_rmse = get_model_rmse(X_goal, groundtruth)
print(f"RMSE of SE model:  {get_model_rmse(X_normal, groundtruth):.4f}")
print(f"RMSE of goal model:  {get_model_rmse(X_goal, groundtruth):.4f}")

print("Saved debugging vectors and matrices to folder Debugging")



