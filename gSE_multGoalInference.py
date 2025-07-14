import numpy as np
import os
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
from dataFunctions import make_groundtruth, pretty_print_matrix, save_vector_arrays_txt, save_matrix_arrays_txt, save_state_comparison_txt, get_model_rmse, save_tracking_plot, extract_params_from_header, save_specifications_txt, save_variance_array_txt
from tqdm import tqdm
from scipy.stats import multivariate_normal

groundtruth_folder = "Data/Generated/Quad/"
debugging_folder = "Debugging/SE_MultTargets/Quad/"
UZH=False
#Model Notes
notes = "Testing prediction of goal from options"

goal_options = [np.array([-20,20]), np.array([20,20]), np.array([20,-20]), np.array([-20,-20])]
goal_indices = [0,1,2,3]
goal_probs = [0.25, 0.25, 0.25, 0.25]



#Common parameters
s2 = 1000
ls = 30
d = 5
dt = 1
sigma_y = 1.0

t = dt * np.arange(d,0,-1)
G_var = 10 #Slightly lower variance for all of them
sigma_g = 0.5
initialize_with_truth = True

#Grid Search
sigma_g_values = [0.1, 0.5, 1.0, 2.0, 5.0]
G_var_values = [10, 100, 1000, 10000]


for subfolder in os.listdir("Data/Generated/Quad/"):
    print(subfolder)
    #Go through each of the quad groundtruths
    groundtruth_path = f"Data/Generated/Quad/{subfolder}/track.txt"
    groundtruth = make_groundtruth(groundtruth_path, UZH=UZH)
    Tmax = groundtruth.shape[0]
    debugging_folder = f"Debugging/SE_MultTargets/Quad/{subfolder}"
    
    TRUE_GOAL = groundtruth[-1,:]

    for sigma_g in sigma_g_values:
        for G_var in G_var_values:
            debugging_folder = f"{debugging_folder}/SG:{sigma_g}/Gvar:{G_var}"

            #Track initialization
            if initialize_with_truth:
                init_point = groundtruth[0,:]
            else:
                init_point = np.array([50,50])

            ##Add noise to create simulated sensor measurements
            noisy_data = [groundtruth[k] + np.random.normal(0, sigma_y, 2) for k in range(Tmax)]

            #Initialize state mean and covariance for EACH goal option
            mk_goals = {}
            vk_goals = {}

            for i, goal_option in enumerate(goal_options):
                name = f"goal_{i}" 
                mk_goal = [init_point * np.ones([d, 2])]
                mk_goal[0] = np.vstack((mk_goal[0], goal_option))

                vk_goal = [np.eye(d+1)]
                vk_goal[0][:-1, :-1] = f.SE(t, t, s2/10, ls)
                vk_goal[0][-1, -1] = G_var

                mk_goals[name] = mk_goal
                vk_goals[name] = vk_goal

            X_goals = {}
            S_goals = {}
            G_goals = {}
            S_goal_vars = {}

            for i, goal_option in enumerate(goal_options):
                name = f"goal_{i}"
                X_goal = np.zeros([Tmax,2])
                S_goal = np.zeros([Tmax]) #Uncertainty at each time step for goal model
                G_goal = np.zeros([Tmax,2])
                S_goal_var = np.zeros([Tmax])

                X_goals[name] = X_goal
                S_goals[name] = S_goal
                G_goals[name] = G_goal
                S_goal_vars[name] = S_goal_var

            likelihoods = {}
            for i, goal_option in enumerate(goal_options):
                name = f"goal_{i}"
                likelihoods[name] = np.zeros([Tmax])

            best_goal = np.zeros([Tmax,2])
            all_likelihoods = np.zeros([Tmax, len(goal_options)])

            goal_estimates_all = np.zeros([4, Tmax, 2])
            goal_vars_all = np.zeros([4, Tmax, 1])

            ##Actual Tracking Portion##
            #For each time step, complete Kalman Filter for each goal-priored state##
            for k in range(Tmax):
                for i, goal_option in enumerate(goal_options):
                    name = f"goal_{i}"
                    mk_current = mk_goals[name][-1]
                    vk_current = vk_goals[name][-1]

                    m_pred, v_pred, F_goal, P_goal = iF.g_se_pred(t,mk_current,vk_current,s2,ls, sigma_g)

                    y = noisy_data[k]
                    datum = y

                    m_up, v_up, KGN, y_in= iF.g_update(datum, m_pred, v_pred, sigma_y)

                    mk_goals[name].append(m_up)
                    vk_goals[name].append(v_up)

                    X_goals[name][k,:] = m_up[0,:] + m_up[-1,:]
                    S_goals[name][k] = v_up[0,0] + v_up[-1,-1]
                    G_goals[name][k,:] = m_up[-1,:]
                    S_goal_vars[name][k] = v_up[-1,-1]

                    cur_pred_loc = m_up[0,:] + m_up[-1,:]
                    cur_pred_cov = v_up[0,0] + v_up[-1,-1]

                    #likelihood = multivariate_normal.pdf(y, cur_pred_loc, cur_pred_cov)
                    #TRYING TO DO LIKELIHOOD THE CURRENT GOAL PREDICTION COMES FROM GOAL OPTIONS#
                    # diff = m_up[-1,:] - goal_options[i]
                    # R = G_var * np.eye(2)
                    # likelihood = -0.5 * diff.T @ np.linalg.inv(R) @ diff
                    # norm_const = 1/ np.sqrt((2*np.pi)**2 * np.linalg.det(R))
                    # likelihood = norm_const * np.exp(likelihood)

                    #MLE for Each Goal Option
                    # cur_best_goal = np.zeros([2])
                    # cur_best_likelihood = -0.1
                    # for i, goal_option in enumerate(goal_options):
                    #     diff = m_up[-1,:] - goal_options[i]
                    #     R = v_up[-1,-1] * np.eye(2)
                    #     likelihood = -0.5 * diff.T @ np.linalg.inv(R) @ diff
                    #     norm_const = 1/ np.sqrt((2*np.pi)**2 * np.linalg.det(R))
                    #     likelihood = norm_const * np.exp(likelihood)
                    #     if likelihood > cur_best_likelihood:
                    #         cur_best_likelihood = likelihood
                    #         cur_best_goal = goal_options[i]

                    # goal_estimates_all[i,k,:] = cur_best_goal

                    ##Using Mahalanobis Distance## NOT THE ACTUAL LIKELIHOOD BUT JUST NAMED FOR SIMPLICITY RN
                    distances = [np.linalg.norm(m_up[-1,:] - g) for g in goal_options]
                    best_goal_idx = np.argmin(distances)
                    goal_estimates_all[i, k, :] = best_goal_idx
                    goal_vars_all[i, k, :] = v_up[-1,-1]
                    #End using Mahalanobis Distance

                    # cur_likelihoods[i] = likelihood
                    # all_likelihoods[k,i] = likelihood**2
                    # print("Current goal particle is:", goal_options[i])
                    # print("Current goal prediction is:", m_up[-1,:])
                    # print("Current likelihood is:", likelihood)

                
                #Calculate the best goal index
                #best_goal[k] = goal_options[np.argmin(cur_likelihoods)]
            all_rmse = {}
            for i, goal_option in enumerate(goal_options):
                name = f"goal_{i}"
                X_goal = X_goals[name]
                G_goal = G_goals[name]
                save_tracking_plot(groundtruth, noisy_data, X_goal, G_goal, X_goal, "Goal-SE", "SE", f"Goal_{i}.png", debugging_folder, show_Target=True, false_goals=goal_options[:3])
                cur_rmse = get_model_rmse(X_goal, groundtruth)
                all_rmse[name] = cur_rmse

            # Save RMSE values to text file
            with open(os.path.join(debugging_folder, "rmse_results.txt"), 'w') as file_handle:
                file_handle.write("RMSE Values by Model\n")
                file_handle.write("=" * 30 + "\n\n")
                for model_name, rmse_value in all_rmse.items():
                    file_handle.write(f"{model_name}: {rmse_value:.6f}\n")
                mean_rmse = np.mean(list(all_rmse.values()))
                file_handle.write(f"\nMean RMSE: {mean_rmse:.6f}\n")
                file_handle.write(f"Std RMSE: {np.std(list(all_rmse.values())):.6f}\n")
                
                # Calculate average end distance
                dists = 0.0
                for i, goal_option in enumerate(goal_options):
                    end_dist = np.linalg.norm(G_goals[f'goal_{i}'][-1,:] - TRUE_GOAL)
                    file_handle.write(f"End Goal for model {i}: {G_goals[f'goal_{i}'][-1,:]}\n")
                    dists += end_dist
                avg_end_distance = dists/4
                file_handle.write(f"Average End Distance: {avg_end_distance:.6f}\n")

                #Calculate RMSE of goal estimates for last 20 time steps
                groundtruth_goal = np.array([-20,-20])
                last_20_steps = min(20, Tmax)  # Use last 20 steps or all steps if Tmax < 20
                
                # Create array of true goal repeated for last 20 time steps
                true_goals_last_20 = np.tile(groundtruth_goal, (last_20_steps, 1))  # Shape: (20, 2)
                
                # Get predicted goals for last 20 time steps for each model
                goal_rmse_scores = []
                for i, goal_option in enumerate(goal_options):
                    predicted_goals_last_20 = G_goals[f'goal_{i}'][-last_20_steps:, :]  # Shape: (20, 2)
                    goal_rmse = np.sqrt(np.mean(np.sum((predicted_goals_last_20 - true_goals_last_20)**2, axis=1)))
                    goal_rmse_scores.append(goal_rmse)
                    file_handle.write(f"Goal RMSE for model {i}: {goal_rmse:.6f}\n")
                
                avg_goal_rmse = np.mean(goal_rmse_scores)
                file_handle.write(f"Average Goal RMSE (last {last_20_steps} steps): {avg_goal_rmse:.6f}\n")
                
                # Calculate combined score (lower is better)
                # Normalize both metrics to 0-1 range and combine with equal weight
                # For now, use reasonable bounds - you can adjust these based on your data
                rmse_normalized = min(mean_rmse / 10.0, 1.0)  # Normalize RMSE (assume max reasonable RMSE is 10)
                distance_normalized = min(avg_end_distance / 50.0, 1.0)  # Normalize distance (assume max reasonable distance is 50)
                
                combined_score = 0.5 * rmse_normalized + 0.5 * distance_normalized
                file_handle.write(f"Combined Score: {combined_score:.6f}\n")
                file_handle.write(f"  (RMSE norm: {rmse_normalized:.6f}, Distance norm: {distance_normalized:.6f})\n")

            np.savetxt(os.path.join(debugging_folder, "best_goals_-20_20.txt"), goal_estimates_all[0, :, :], fmt='%d')
            np.savetxt(os.path.join(debugging_folder, "best_goals_20_20.txt"), goal_estimates_all[1, :, :], fmt='%d')
            np.savetxt(os.path.join(debugging_folder, "best_goals_20_-20.txt"), goal_estimates_all[2, :, :], fmt='%d')
            np.savetxt(os.path.join(debugging_folder, "best_goals_-20_-20.txt"), goal_estimates_all[3, :, :], fmt='%d')
            for i in range(4):
                np.savetxt(os.path.join(debugging_folder, f"variances_goal_{i}.txt"), S_goal_vars[f"goal_{i}"], fmt='%.6f')











