import numpy as np
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
from dataFunctions import make_groundtruth, pretty_print_matrix, save_vector_arrays_txt, save_matrix_arrays_txt, save_state_comparison_txt, get_model_rmse, save_tracking_plot
from tqdm import tqdm

#groundtruth = np.load('ground_truth_trajectory.npy') #This is fake for the moment, obviously
## For UZH Data Test ##
# groundtruth = make_groundtruth("Data/groundtruth.txt", UZH=True)
# groundtruth = groundtruth[::20]

## For Generated Data ##
groundtruth = make_groundtruth("Data/goalMeanGP_track.txt", UZH=False)

#Common parameters portion
######
d=5
dt = 1 #Just assumed for now
Tmax = groundtruth.shape[0]
t = dt * np.arange(d,0,-1)
s2 = 100
ls = 3
debugging_folder = "Debugging/goalMeanGPGenData_SEfromFunctions"

G_prior = np.array([[10.0, 10.0]])
G_var = 1  #Unsure goal initially

##Adding noise to create simulated sensor measurements if only groundtruth data available
sy = 0.5
noisy_data = [groundtruth[k] + np.random.normal(0, sy, 2) for k in range(Tmax)]

#Initialization portion
######
mk_normal = [groundtruth[0,:]*np.ones([d,2])]
vk_normal = [f.SE(t,t,s2/10,ls)]

mk_goal = [groundtruth[0,:]*np.ones([d,2])]
mk_goal[0] = np.vstack((mk_goal[0], G_prior))
#Making covariance matrix for goal model
vk_goal = [np.eye(d+1)]
vk_goal[0][:-1,:-1] = f.SE(t,t,s2/10,ls)
vk_goal[0][-1,-1] = G_var

#Final objects
######
X_normal = np.zeros([Tmax,2])
S_normal = np.zeros([Tmax]) #Uncertainty at each time step

X_goal = np.zeros([Tmax,2])
S_goal = np.zeros([Tmax]) #Uncertainty at each time step for goal model
G_goal = np.zeros([Tmax,2])

#Debugging objects
normal_F_aug = []
goal_F_aug = []

normal_predicted_means = []
goal_predicted_means = []
normal_updated_means = []
goal_updated_means = []

##Tracking portion
##Normal GP (SE) model
for k in range(Tmax):
    #Prediction step
    m_predN, v_predN, F_augN = f.se_pred(t,mk_normal[-1],vk_normal[-1],s2,ls)

    #Skip association step for now
    y = noisy_data[k]
    datum = y

    #Update step
    m_upN, v_upN, KGN, y_inN = f.update(datum, m_predN, v_predN, sy)

    mk_normal.append(m_upN)
    vk_normal.append(v_upN)

    X_normal[k,:] = m_upN[0,:] + m_upN[-1,:]
    S_normal[k] = v_upN[0,0] + v_upN[-1,-1] 

    normal_F_aug.append(F_augN.copy())
    normal_predicted_means.append(m_predN.copy())
    normal_updated_means.append(m_upN.copy())
#Extended Goal Model
for k in range(Tmax):
    m_pred, v_pred, F_goal = iF.g_se_pred(t,mk_goal[-1],vk_goal[-1],s2,ls)

    y = noisy_data[k]
    datum = y

    m_up, v_up, KGN, y_in = iF.g_update(datum, m_pred, v_pred, sy)

    mk_goal.append(m_up)
    vk_goal.append(v_up)
    X_goal[k,:] = m_up[0,:] + m_up[-2,:] #Measurement model dependent only on initial location
    X_goal[k,:] = m_up[0,:] + m_up[-2,:] + m_up[-1,:]  #Measurement model dependent on goal
    S_goal[k] = v_up[0,0] + v_up[-2,-2] 

    G_goal[k,:] = m_up[-1,:]

    goal_F_aug.append(F_goal.copy())
    goal_predicted_means.append(m_pred.copy())
    goal_updated_means.append(m_up.copy())

#Save results for debugging
save_tracking_plot(groundtruth, noisy_data, X_goal, G_goal, X_normal, "Goal-SE", "SE", "normalGPwithGoal_UZH3.png", debugging_folder)
save_matrix_arrays_txt(normal_F_aug[:5], goal_F_aug[:5], "normalGP_transitionMatrices_UZH3.txt", "F_aug", "F_goal", debugging_folder)
save_vector_arrays_txt(normal_predicted_means, goal_predicted_means, "normalGP_predictedMeans_UZH3.txt", "m_predN", "m_pred", debugging_folder)
save_vector_arrays_txt(normal_updated_means, goal_updated_means, "normalGP_updatedMeans_UZH3.txt", "m_updN", "m_upd", debugging_folder)

print(f"RMSE of SE model:  {get_model_rmse(X_normal, groundtruth):.4f}")
print(f"RMSE of goal model:  {get_model_rmse(X_goal, groundtruth):.4f}")

print("Saved debugging vectors and matrices to folder Debugging")






