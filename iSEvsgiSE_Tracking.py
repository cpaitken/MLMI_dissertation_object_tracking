import numpy as np
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
from dataFunctions import make_groundtruth, pretty_print_matrix, save_vector_arrays_txt, save_matrix_arrays_txt, save_state_comparison_txt, get_model_rmse, save_tracking_plot, save_variance_array_txt
from tqdm import tqdm
import random
random.seed(24)
np.random.seed(24)


groundtruth_filename = "Data/Generated/Simple_iSE_Track/goal_iSE_track.txt"
debugging_folder = "Debugging/iSE/BasicMeasurementModel/Simple_iSE_Track"
UZH = False
#groundtruth = groundtruth[::20]
groundtruth = make_groundtruth(groundtruth_filename, UZH=UZH)
if UZH:
    groundtruth = groundtruth[::20]

## Hyperparameter specification ##
d = 5 #Sliding window
s2 = 10 #Prior output variance
ls = 5 #Length scale
T = groundtruth.shape[0]
Tmax = T #Maximum steps
dt = 1
t = dt * np.arange(d, 0, -1) #Time vector with most recent measurement first
assoc_threshold = 5
initializeGoal_with_truth = True



##Adding noise to create simulated sensor measurements if only groundtruth data available
sy = 1.0
noisy_data = [groundtruth[k] + np.random.normal(0, sy, 2) for k in range(Tmax)]


## Goal Creation ##
if initializeGoal_with_truth:
    G_prior = np.array([50,100])
else:
    G_prior = np.array([0,0])
G_var = 10000  #Unsure goal initially

################################
##Initialization of iSE Model##
## Same as Final_Double_Swap.py ##
################################
mk_normal= [groundtruth[0, :] * np.ones([d, 2])]
mk_normal[0][:-1] -= mk_normal[0][-1]  # iSE-1 style offset
vk_normal = [np.eye(d)]
vk_normal[0][:-1,:-1] = f.iSE(t[1:],t[1:],s2/10,ls)
vk_normal[0][-1,-1] = s2/10

################################
##Initialization of goal-iSE Model##
################################
mk_goal = [groundtruth[0, :] * np.ones([d, 2])]
mk_goal[0][:-1] -= mk_goal[0][-1]
mk_goal[0] = np.vstack((mk_goal[0], G_prior))

#Create Equivalent to vk_normal for upper left block#
P_goal = np.zeros((d+1, d+1))
P_goal[:d-1, :d-1] = f.iSE(t[1:], t[1:], s2/10, ls)
P_goal[-2, -2] = s2/10
P_goal[-1,-1] = G_var
##Experimental Section - Try to add correlation between most recent state and goal ##
P_goal[0,-1] = 0.00
P_goal[-1,0] = 0.00
##End Experimental Section##
vk_goal = [P_goal]

################################
##Final objects##
################################
X_normal = np.zeros([Tmax,2]) #Keep track of predicted state
S_normal = np.zeros([Tmax])
#Infered Goal Model Storage
X_goal = np.zeros([Tmax,2]) #Keep track of predicted state
S_goal = np.zeros([Tmax])
G_goal = np.zeros([Tmax, 2]) #Keep track of predicted goal
S_goal_var = np.zeros([Tmax])

#Storage for full debugging vectors
normal_F_aug = []
goal_F_aug = []
normal_Covar = []
goal_Covar = []

normal_predicted_means = []
goal_predicted_means = []
normal_updated_means = []
goal_updated_means = []

########################
##Tracking Portion for iSE Model##
################################
normal_Covar.append(vk_normal[-1].copy())
for k in range(Tmax):
    m_predN, v_predN, F_aug, P_normal = f.ise1_pred(t+dt*(k+1), mk_normal[-1], vk_normal[-1], s2, ls)

    normal_predicted_means.append(m_predN.copy())
    
    

    # One observation only
    y = noisy_data[k]
    datum = y

    # Update
    m_upN, v_upN, KG, y_in = f.update_ise1(datum, m_predN, v_predN, sy)

    # Record
    mk_normal.append(m_upN)
    vk_normal.append(v_upN)

    X_normal[k, :] = m_upN[0, :] + m_upN[-1, :]  # Estimate + inferred goal component
    S_normal[k] = v_upN[0,0] + v_upN[-1,-1]
    
    # Store full state vector for iSE model
    normal_updated_means.append(m_upN.copy()) #m_upN is the same as m_predN
    normal_F_aug.append(F_aug.copy())
    normal_Covar.append(v_predN.copy())
    

########################
##Tracking Portion for goal-iSE Model##
################################
goal_Covar.append(vk_goal[-1].copy())
for k in range(Tmax):
    #Predict using the augmented state space model
    m_pred, v_pred, F_goal, P_goal = iF.gise1_pred(t+dt*(k+1), mk_goal[-1], vk_goal[-1], s2, ls, sigma_g=2)

    #Observation
    y = noisy_data[k]
    datum = y

    #Update with Kalman Filter
    m_up, v_up, KG, y_in = iF.augmented_update(datum, m_pred, v_pred, sy)

    mk_goal.append(m_up)
    vk_goal.append(v_up)

    X_goal[k, :] = m_up[0, :] + m_up[-2, :] + m_up[-1, :] #Predicted most recent location
    S_goal[k] = v_up[0,0] + v_up[-2,-2] + v_up[-1,-1]
    S_goal_var[k] = v_up[-1,-1]#X[k, :] = m_up[0, :] + +m_up[-2, :] + m_up[-1, :] #Predicted most recent location and goal
    G_goal[k, :] = m_up[-1, :] #Predicted goal
    
    # Store full state vector for goal model
    goal_F_aug.append(F_goal.copy())
    goal_Covar.append(v_pred.copy())
    goal_predicted_means.append(m_pred.copy())
    goal_updated_means.append(m_up.copy())

save_tracking_plot(groundtruth, noisy_data, X_goal, G_goal, "Goal-iSE", "ComparisonPlot.png", debugging_folder, show_Target=False, true_goal=G_prior, false_goals=None, XN=X_normal, modelName2="iSE")
save_matrix_arrays_txt(normal_F_aug[:5], goal_F_aug[:5], "transitionMatrices.txt", "F_aug", "F_goal", debugging_folder)
save_matrix_arrays_txt(normal_Covar[:5], goal_Covar[:5], "covarianceMatrices.txt", "Covar Normal", "Covar Goal", debugging_folder)
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



