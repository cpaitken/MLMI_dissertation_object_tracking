import numpy as np
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
from dataFunctions import make_groundtruth, pretty_print_matrix, save_vector_arrays_txt, save_matrix_arrays_txt, save_state_comparison_txt, get_model_rmse, save_tracking_plot, extract_params_from_header, save_specifications_txt
from tqdm import tqdm

#groundtruth = np.load('ground_truth_trajectory.npy') #This is fake for the moment, obviously
## THINGS TO ACTUALLY CHANGE ##
groundtruth_filename = "Data/UZH/Medium/UZH_5.txt" ##CHANGE THIS FOR DATA
debugging_folder = "Debugging/UZH/Medium/UZH_5"
UZH=True
#Model Notes
notes = "Set goal prior to be last point in groundtruth"

G_prior = np.array([7.0, -3.0])
G_var = 10000  #Unsure goal initially
initialize_with_truth = True

#Model parameters to change
s2 = 100
ls = 20
d=5



groundtruth = make_groundtruth(groundtruth_filename, UZH=UZH)
if UZH:
    groundtruth = groundtruth[::20]

data_params = extract_params_from_header(groundtruth_filename)

if initialize_with_truth:
    init_point = groundtruth[0,:]
else:
    init_point = np.array([50,50])


# Dictionary to store model parameters for saving
model_params = {
    'MODEL_G_prior': G_prior.tolist(),
    'MODEL_G_var': G_var,
    'MODEL_s2': s2,
    'MODEL_ls': ls,
    'MODEL_d': d,
    'MODEL_start_point': init_point.tolist(),
    'MODEL_notes': notes
}


#Common parameters portion
######
dt = 1 #Just assumed for now
Tmax = groundtruth.shape[0]
t = dt * np.arange(d,0,-1)

##Adding noise to create simulated sensor measurements if only groundtruth data available
sy = 1
noisy_data = [groundtruth[k] + np.random.normal(0, sy, 2) for k in range(Tmax)]


#Initialization portion
######
mk_normal = [init_point*np.ones([d,2])]
vk_normal = [f.SE(t,t,s2/10,ls)]

mk_goal = [init_point*np.ones([d,2])]
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
S_goal_var = np.zeros([Tmax])

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

    #X_normal[k,:] = m_upN[0,:] + m_upN[-1,:]
    X_normal[k,:] = m_upN[0,:]
    #S_normal[k] = v_upN[0,0] + v_upN[-1,-1] 
    S_normal[k] = v_upN[0,0]

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
    #X_goal[k,:] = m_up[0,:] + m_up[-2,:] #Measurement model dependent only on initial location
    #X_goal[k,:] = m_up[0,:] + m_up[-2,:] + m_up[-1,:]  #Measurement model dependent on goal
    X_goal[k,:] = m_up[0,:] + m_up[-1,:]
    #S_goal[k] = v_up[0,0] + v_up[-2,-2] 
    S_goal[k] = v_up[0,0] + v_up[-1,-1] #For the predicted location
    S_goal_var[k] = v_up[-1,-1] #For just the goal itself

    G_goal[k,:] = m_up[-1,:]

    goal_F_aug.append(F_goal.copy())
    goal_predicted_means.append(m_pred.copy())
    goal_updated_means.append(m_up.copy())

#Save results for debugging
save_tracking_plot(groundtruth, noisy_data, X_goal, G_goal, X_normal, "Goal-SE", "SE", "ComparisonPlot.png", debugging_folder)
save_matrix_arrays_txt(normal_F_aug[:5], goal_F_aug[:5], "transitionMatrices.txt", "F_aug", "F_goal", debugging_folder)
save_vector_arrays_txt(normal_predicted_means, goal_predicted_means, "predictedMeans.txt", "m_predN", "m_pred", debugging_folder)
save_vector_arrays_txt(normal_updated_means, goal_updated_means, "updatedMeans.txt", "m_updN", "m_upd", debugging_folder)
save_variance_array_txt(S_goal_var, "goalVariances.txt", debugging_folder)
save_variance_array_txt(S_normal, "normalLocationVariances.txt", debugging_folder)
save_variance_array_txt(S_goal, "goalLocationVariances.txt", debugging_folder)

se_rmse = get_model_rmse(X_normal, groundtruth)
gse_rmse = get_model_rmse(X_goal, groundtruth)
performance_params = {
    "SE_RMSE": se_rmse,
    "GSE_RMSE": gse_rmse
}

all_params = {
    "dataset_params": data_params,
    "model_params": model_params,
    "performance_params": performance_params
}
save_specifications_txt(debugging_folder, all_params)

#See if plotting is the issue
# plt.figure(figsize=(8, 6))
# plt.scatter(*zip(*noisy_data), alpha=0.3, label='Noisy obs')
# plt.plot(groundtruth[:, 0], groundtruth[:, 1], 'g-', label='Groundtruth', linewidth=2)
# plt.plot(X_normal[:, 0], X_normal[:, 1], 'r-', label='Kalman Filter', linewidth=2)
# plt.plot(X_goal[:, 0], X_goal[:, 1], 'p-', label='Kalman Filter Goal Aware', linewidth=2)
# plt.legend()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('2D Trajectory')
# plt.axis('equal')
# plt.show()

print(f"RMSE of SE model:  {get_model_rmse(X_normal, groundtruth):.4f}")
print(f"RMSE of goal model:  {get_model_rmse(X_goal, groundtruth):.4f}")

print("Saved debugging vectors and matrices to folder Debugging")






