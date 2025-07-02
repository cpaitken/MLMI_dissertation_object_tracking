import numpy as np
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
from dataFunctions import make_groundtruth, pretty_print_matrix, save_vector_arrays_txt, save_matrix_arrays_txt, save_state_comparison_txt, get_model_rmse, save_tracking_plot, extract_params_from_header, save_specifications_txt, save_variance_array_txt
from tqdm import tqdm
np.random.seed(24)

## TRACKING FUNCTION DEFINITIONS ##
#Definition for tracking SE model
def track_SE(Tmax, s2, ls, sy, mk_normal, vk_normal, noisy_data, X_normal, S_normal, normal_F_aug, normal_predicted_means, normal_updated_means):
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


        X_normal[k,:] = m_upN[0,:]
        S_normal[k] = v_upN[0,0]

        normal_F_aug.append(F_augN.copy())
        normal_predicted_means.append(m_predN.copy())
        normal_updated_means.append(m_upN.copy())

    return X_normal, S_normal, normal_F_aug, normal_predicted_means, normal_updated_means

def track_GSE(Tmax, s2, ls, sy, mk_goal, vk_goal, noisy_data, X_goal, S_goal, G_goal, S_goal_var, goal_F_aug, goal_predicted_means, goal_updated_means):
    for k in range(Tmax):
        m_pred, v_pred, F_goal = iF.g_se_pred(t,mk_goal[-1],vk_goal[-1],s2,ls)

        y = noisy_data[k]
        datum = y

        m_up, v_up, KGN, y_in = iF.g_update(datum, m_pred, v_pred, sy)

        mk_goal.append(m_up)
        vk_goal.append(v_up)
        
        X_goal[k,:] = m_up[0,:] + m_up[-1,:]
        S_goal[k] = v_up[0,0] + v_up[-1,-1] #For the predicted location
        S_goal_var[k] = v_up[-1,-1] #For just the goal itself
        G_goal[k,:] = m_up[-1,:]

        goal_F_aug.append(F_goal.copy())
        goal_predicted_means.append(m_pred.copy())
        goal_updated_means.append(m_up.copy())

    return X_goal, S_goal, G_goal, S_goal_var, goal_F_aug, goal_predicted_means, goal_updated_means

#Define all the files to go through
Easy = ["UZH_1", "UZH_2", "UZH_3", "UZH_4", "UZH_9", "UZH_10", "UZH_12"]
Medium = ["UZH_1", "UZH_3", "UZH_5", "UZH_6", "UZH_9", "UZH_13"]
Hard = ["UZH_5", "UZH_7", "UZH_14"]

# Easy = ["UZH_3"]
# Medium = ["UZH_1"]
# Hard = ["UZH_5"]

#Define the folders to go through
All_categories = [Easy, Medium, Hard]
category_names = ["Easy", "Medium", "Hard"]

#Define the common parameters for both models
s2 = 100
ls = 20
d = 5
dt = 1
sy = 1
G_var = 10000
initializeTrack_with_truth = True
initializeGoal_with_truth = False
UZH = True
run_name = "Goal_False_Init"


#Call tracking for SE model on all files
for category, category_name in zip(All_categories, category_names):
    for file in category: #For each UZH file
        #Make the groundtruth
        groundtruth_filename = f"Data/UZH/{category_name}/{file}.txt"
        groundtruth = make_groundtruth(groundtruth_filename, UZH=UZH)
        if UZH:
            groundtruth = groundtruth[::20]
        data_params = extract_params_from_header(groundtruth_filename)

        #Make debugging folder
        debugging_folder = f"Debugging/UZH/{run_name}/{category_name}/{file}"

        if initializeTrack_with_truth:
            init_point = groundtruth[0,:]
        else:
            init_point = np.array([50,50])
        
        #Set goal for goal model
        if initializeGoal_with_truth:
            G_prior = groundtruth[-1,:]
        else:
            G_prior = np.array([50,50])

        #Common parameters dependent on the file
        Tmax = groundtruth.shape[0]
        t = dt * np.arange(d,0,-1)

        #Make noisy data for both models
        noisy_data = [groundtruth[k] + np.random.normal(0, sy, 2) for k in range(Tmax)]

        #Initialize the SE Mean and Covariance
        mk_normal = [init_point*np.ones([d,2])]
        vk_normal = [f.SE(t,t,s2/10,ls)]

        #Initialize GSE Mean and Covariance
        mk_goal = [init_point*np.ones([d,2])]
        mk_goal[0] = np.vstack((mk_goal[0], G_prior))

        vk_goal = [np.eye(d+1)]
        vk_goal[0][:-1,:-1] = f.SE(t,t,s2/10,ls)
        vk_goal[0][-1,-1] = G_var
        
        #Make final objects
        X_normal = np.zeros([Tmax,2])
        S_normal = np.zeros([Tmax]) #Uncertainty at each time step

        X_goal = np.zeros([Tmax,2])
        S_goal = np.zeros([Tmax]) #Uncertainty at each time step for goal model
        G_goal = np.zeros([Tmax,2])
        S_goal_var = np.zeros([Tmax])

        #Make debugging and specification objects
        normal_F_aug = []
        goal_F_aug = []
        normal_predicted_means = []
        goal_predicted_means = []
        normal_updated_means = []
        goal_updated_means = []

        model_params = {
            'MODEL_G_prior': G_prior.tolist(),
            'MODEL_G_var': G_var,
            'MODEL_s2': s2,
            'MODEL_ls': ls,
            'MODEL_d': d,
            'MODEL_start_point': init_point.tolist(),
            "Track_with_truth": initializeTrack_with_truth,
            "Goal_with_truth": initializeGoal_with_truth
        }

        #Call tracking for SE and return all the results
        X_normal, S_normal, normal_F_aug, normal_predicted_means, normal_updated_means = track_SE(Tmax, s2, ls, sy, mk_normal, vk_normal, noisy_data, X_normal, S_normal, normal_F_aug, normal_predicted_means, normal_updated_means)
        X_goal, S_goal, G_goal, S_goal_var, goal_F_aug, goal_predicted_means, goal_updated_means = track_GSE(Tmax, s2, ls, sy, mk_goal, vk_goal, noisy_data, X_goal, S_goal, G_goal, S_goal_var, goal_F_aug, goal_predicted_means, goal_updated_means)

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


