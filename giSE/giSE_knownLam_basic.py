##File to test the g-iSE model on goal-converging trajectories (one lambda_gen value dataset and initialisation combination at a time) ##
## Change lambda_gen dataset used in line 18 ##
## Change initialisation combinations in lines 28 and 29 ##
## Currently runs for lambda_gen = 0.05, initialisation with true goal and lambda ##

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
from dataFunctions import make_groundtruth, get_model_rmse, save_tracking_plot, save_vector_arrays_txt, save_matrix_arrays_txt
import random
random.seed(24)
np.random.seed(24)

lamVal=0.05
groundtruth_folder = f"Data/Generated/giSE_convergingMeasModel/constantLambda/Tmax100/lambda_{lamVal}"
debugging_folder = f"Debugging/conv_iSE/constantLambda/Tmax_100_FINAL/lambda_{lamVal}/FalseGoal_TrueLambda"
true_goal = np.array([50.0, 50.0])
false_goal = np.array([25.0, 25.0])
true_lambda = lamVal
false_lambda = true_lambda + 0.01

################################
## Choices to make ##
initialize_goal_with_truth = True
initialize_lambda_with_truth = True

sigma_g_options = [0, 2.5, 5, 10, 20, 30]
G_var_options = [0, 50, 100]
################################


################################
## Common Model Parameters ##
## Same as generating Model ##
d=5
s2 = 10
ls = 5
sy=0.25

t = np.arange(d,0,-1)
dt=1

if initialize_goal_with_truth:
    G_prior = true_goal
else:
    G_prior = false_goal
#G_var = 0
#sigma_g = 20


if initialize_lambda_with_truth:
    lambda_val = true_lambda
else:
    lambda_val = false_lambda

##Make the noisy data with very small noise##
# sy=0.5
# noisy_data = [groundtruth[k] + np.random.normal(0, sy, 2) for k in range(Tmax)]


################################
##Initialization for normaliSE##
################################
# mk_normal = [groundtruth[0, :] * np.ones([d, 2])]
# mk_normal[0][:-1] -= mk_normal[0][-1]

# vk_normal = [np.eye(d)]
# vk_normal[0][:-1,:-1] = f.iSE(t[1:],t[1:],s2/10,ls)
# vk_normal[0][-1,-1] = s2/10
# ################################

# ################################
# ##Initialization for g-iSE##
# ################################
# mk_goal = [groundtruth[0, :] * np.ones([d, 2])]
# mk_goal[0][:-1] -= mk_goal[0][-1]
# mk_goal[0] = np.vstack((mk_goal[0], G_prior))

# P_goal = np.zeros((d+1, d+1))
# P_goal[:d-1, :d-1] = f.iSE(t[1:], t[1:], s2/10, ls)
# P_goal[-2, -2] = s2/10
# P_goal[-1,-1] = G_var

# vk_goal = [P_goal]
# ################################

# ################################
# ##Final objects##
# ################################
# X_normal = np.zeros([Tmax,2]) #Keep track of predicted state
# S_normal = np.zeros([Tmax])

# X_goal = np.zeros([Tmax,2]) #Keep track of predicted state
# S_goal = np.zeros([Tmax])
# G_goal = np.zeros([Tmax, 2]) #Keep track of predicted goal
# S_goal_var = np.zeros([Tmax])
# ################################

# ################################
# ##Debugging Objects##
# predicted_means_goal = []
# updated_means_goal = []
# predicted_locations_goal = []
# decay_rates_kF = []
# decay_rates_tracking = []
# transition_matrices_goal = []
# covariance_matrices_goal = []
# predicted_locations_atTheUpdateStep = []

# predicted_means_normal = []
# updated_means_normal = []
# predicted_locations_normal = []
# transition_matrices_normal = []
# covariance_matrices_normal = []
################################

################################
## Dictionary for each parameter combination RMSE ##
param_combo_rmse = {}
param_combo_change_to_normal = {}
param_combo_distance_to_goal = {}
order_of_files = []

for sigma_g in sigma_g_options:
    for G_var in G_var_options:
        param_combo_rmse[f"SG:{sigma_g}_GV:{G_var}"] = []
        param_combo_change_to_normal[f"SG:{sigma_g}_GV:{G_var}"] = []
        param_combo_distance_to_goal[f"SG:{sigma_g}_GV:{G_var}"] = []
################################

################################
##Tracking Portion for normal iSE##
## Group all individual trajectory folders into one big folder ##
ind_runs_folder = os.path.join(debugging_folder, "individual_runs")
os.makedirs(ind_runs_folder, exist_ok=True)

for groundtruth_filename in os.listdir(groundtruth_folder):
    print(f"Current file: {groundtruth_filename}")
    if groundtruth_filename.endswith(".png"):
        print("Skipping png file")
        continue
    ##Make the groundtruth for this specific file##
    
    groundtruth = make_groundtruth(os.path.join(groundtruth_folder, groundtruth_filename))
    Tmax = groundtruth.shape[0]
    noisy_data = [groundtruth[k] + np.random.normal(0, sy, 2) for k in range(Tmax)]
    ################################################

    ##Make the debugging folder for this specific file##
    filename_no_ext = groundtruth_filename.replace(".txt", "")
    track_specific_folder = os.path.join(ind_runs_folder, filename_no_ext[-1])
    order_of_files.append(filename_no_ext[-1])
    os.makedirs(track_specific_folder, exist_ok=True)

    ##Go through each sigma_g ##
    for sigma_g in sigma_g_options:
        sigma_g_folder = os.path.join(track_specific_folder, f"SG:{sigma_g}")
        os.makedirs(sigma_g_folder, exist_ok=True)

        ## Go through each G_Var ##
        for G_var in G_var_options:
            G_var_folder = os.path.join(sigma_g_folder, f"GV:{G_var}")
            os.makedirs(G_var_folder, exist_ok=True)

            ## Set up dictionary for parameter combination ##
            param_combo_name = f"SG:{sigma_g}_GV:{G_var}"


            #### Reset Initialization Objects and Final Objects ##
            ################################
            ##Initialization for normaliSE##
            ################################
            mk_normal = [groundtruth[0, :] * np.ones([d, 2])]
            mk_normal[0][:-1] -= mk_normal[0][-1]

            vk_normal = [np.eye(d)]
            vk_normal[0][:-1,:-1] = f.iSE(t[1:],t[1:],s2/10,ls)
            vk_normal[0][-1,-1] = s2/10
            ################################

            ################################
            ##Initialization for g-iSE##
            ################################
            mk_goal = [groundtruth[0, :] * np.ones([d, 2])]
            mk_goal[0][:-1] -= mk_goal[0][-1]
            mk_goal[0] = np.vstack((mk_goal[0], G_prior))

            P_goal = np.zeros((d+1, d+1))
            P_goal[:d-1, :d-1] = f.iSE(t[1:], t[1:], s2/10, ls)
            P_goal[-2, -2] = s2/10
            P_goal[-1,-1] = G_var

            vk_goal = [P_goal]
            ################################

            ################################
            ##Final objects##
            ################################
            X_normal = np.zeros([Tmax,2]) #Keep track of predicted state
            S_normal = np.zeros([Tmax])

            X_goal = np.zeros([Tmax,2]) #Keep track of predicted state
            S_goal = np.zeros([Tmax])
            G_goal = np.zeros([Tmax, 2]) #Keep track of predicted goal
            S_goal_var = np.zeros([Tmax])
            ################################

            ################################
            ##Debugging Objects##
            predicted_means_goal = []
            updated_means_goal = []
            predicted_locations_goal = []
            decay_rates_kF = []
            decay_rates_tracking = []
            transition_matrices_goal = []
            covariance_matrices_goal = []
            predicted_locations_atTheUpdateStep = []

            predicted_means_normal = []
            updated_means_normal = []
            predicted_locations_normal = []
            transition_matrices_normal = []
            covariance_matrices_normal = []
            ################################

            ## Normal iSE Tracking ##
            for k in range(Tmax):
                m_predN, v_predN, F_normal, P_normal = f.ise1_pred(t+dt*(k+1), mk_normal[-1], vk_normal[-1], s2, ls)
                
                y = noisy_data[k]
                datum = y

                m_upN, v_upN, KG, y_in = f.update_ise1(datum, m_predN, v_predN, sy)

                mk_normal.append(m_upN)
                vk_normal.append(v_upN)

                X_normal[k,:] = m_upN[0,:] + m_upN[-1,:]
                predicted_locations_normal.append(m_upN[0,:] + m_upN[-1,:])
                S_normal[k] = v_upN[0,0] + v_upN[-1,-1]

                predicted_means_normal.append(m_predN.copy())
                updated_means_normal.append(m_upN.copy())
                transition_matrices_normal.append(F_normal.copy())
                covariance_matrices_normal.append(P_normal.copy())
    ################################

    ################################
            ##Tracking Portion for g-iSE##
            for k in range(Tmax):
                m_pred, v_pred, F_goal, P_goal = iF.gise1_pred(t+dt*(k+1), mk_goal[-1], vk_goal[-1], s2, ls, sigma_g=sigma_g)

                y = noisy_data[k]
                datum = y

                m_up, v_up, KG, decay_rate_updateStep, predicted_loc_updateStep = iF.fixed_lambda_kf_update(datum, m_pred, v_pred, sy, k, lambda_val)
                
                mk_goal.append(m_up)
                vk_goal.append(v_up)

                ### Portion for New Measurement Model ##
                decay_rate = np.exp(-1*lambda_val*(k))
                H = np.zeros((1, m_up.shape[0]))
                H[0,0] = decay_rate
                H[0,-2] = decay_rate
                H[0, -1] = (1 - decay_rate)
                pred_location_afterUpdate = H @ m_up.copy()

                ## Adding to the Final Objects ##
                X_goal[k,:] = pred_location_afterUpdate
                predicted_locations_goal.append(pred_location_afterUpdate)
                S_goal[k] = v_up[0,0] + v_up[-2,-2] + v_up[-1,-1]
                S_goal_var[k] = v_up[-1,-1]
                G_goal[k,:] = m_up[-1,:]

                predicted_means_goal.append(m_pred.copy())
                updated_means_goal.append(m_up.copy())
                transition_matrices_goal.append(F_goal.copy())
                covariance_matrices_goal.append(P_goal.copy())
                decay_rates_kF.append(decay_rate_updateStep)
                decay_rates_tracking.append(decay_rate)
                predicted_locations_atTheUpdateStep.append(predicted_loc_updateStep)
    ################################

    ################################
    ## Save debug file to folder specifically for that sigma and G_var combo ##
            overall_rmse = get_model_rmse(X_goal, groundtruth)
            normal_rmse = get_model_rmse(X_normal, groundtruth)

            ## Final Result Dictionary ##
            param_combo_rmse[param_combo_name].append(overall_rmse)
            param_combo_change_to_normal[param_combo_name].append(normal_rmse)
            ## Calculate distance to goal -- If goal is not the true goal, then calculate distance to true goal ##
            param_combo_distance_to_goal[param_combo_name].append(np.linalg.norm(G_goal[-1] - true_goal))
            
            ##############################

            print("Completed tracking for sigma_g:", sigma_g, "and G_var:", G_var)
            save_tracking_plot(groundtruth, noisy_data, X_goal, G_goal, "Goal-iSE", "ComparisonPlot.png", G_var_folder, show_Target=False, true_goal=G_prior, false_goals=None, XN=X_normal, modelName2="iSE")
            #save_vector_arrays_txt(predicted_means_normal, predicted_means_goal, "predictedMeans.txt", "m_predN", "m_pred", debugging_folder)
            #save_vector_arrays_txt(updated_means_normal, updated_means_goal, "updatedMeans.txt", "m_updN", "m_upd", debugging_folder)
            #save_vector_arrays_txt(predicted_locations_normal, predicted_locations_goal, "predictedLocations.txt", "m_predN", "m_pred", debugging_folder)
            #save_matrix_arrays_txt(transition_matrices_normal, transition_matrices_goal, "transitionMatrices.txt", "F_normal", "F_goal", debugging_folder)
            #save_matrix_arrays_txt(covariance_matrices_normal, covariance_matrices_goal, "covarianceMatrices.txt", "P_normal", "P_goal", debugging_folder)

            ##Make overall debugging file##
            with open(os.path.join(G_var_folder, "overall_debugging.txt"), "w") as debug_file:
                debug_file.write("Overall Debugging Information\n")
                debug_file.write("=" * 40 + "\n\n")
                
                for k in range(min(50, Tmax)):
                    debug_file.write(f"Time step {k}:\n")
                    debug_file.write("-" * 20 + "\n")
                    
                    # Normal iSE debugging info
                    debug_file.write(f" NORMAL:Predicted Mean (shape: {predicted_means_normal[k].shape}):\n")
                    for i in range(predicted_means_normal[k].shape[0]):
                        debug_file.write(f"    {i}: {predicted_means_normal[k][i]}\n")
                    debug_file.write(f"GOALED:Predicted Mean (shape: {predicted_means_goal[k].shape}):\n")
                    for i in range(predicted_means_goal[k].shape[0]):
                        debug_file.write(f"    {i}: {predicted_means_goal[k][i]}\n")

                    #Updated Means
                    debug_file.write(f"  Updated Mean (shape: {updated_means_normal[k].shape}):\n")
                    for i in range(updated_means_normal[k].shape[0]):
                        debug_file.write(f"    {i}: {updated_means_normal[k][i]}\n")
                    debug_file.write(f"  Updated Mean (shape: {updated_means_goal[k].shape}):\n")
                    for i in range(updated_means_goal[k].shape[0]):
                        debug_file.write(f"    {i}: {updated_means_goal[k][i]}\n")

                    #Values used in the updated Step
                    debug_file.write(f"  Decay Rate (KF): {decay_rates_kF[k]:.6f}\n")
                    debug_file.write(f"  Predicted Location at Update Step: {predicted_locations_atTheUpdateStep[k]}\n\n")

                    #Making the predicted location
                    debug_file.write(f"GOALED: Decay Rate (Tracking): {decay_rates_tracking[k]:.6f}\n")
                    debug_file.write(f"GOALED: Predicted Location: {predicted_locations_goal[k]}\n")
                    debug_file.write(f"NORMAL: Predicted Location: {predicted_locations_normal[k]}\n")
                    debug_file.write(f"  Predicted Goal: {G_goal[k]}\n")
                    debug_file.write("\n\n")
            
            with open(os.path.join(G_var_folder, "overall_results.txt"), "a") as results_file:
                results_file.write(f"Sigma_g: {sigma_g}, G_var: {G_var}\n\n")
                results_file.write(f"Normal RMSE: {normal_rmse:.6f}\n")
                results_file.write(f"Goal RMSE: {overall_rmse:.6f}\n")
                results_file.write(f"Final predicted goal: {G_goal[-1]}\n\n")
                    
    ################################

## Final Results per Parameter Combination for this group of trajectories ##
with open(os.path.join(debugging_folder, "final_results.txt"), "w") as overall_results_file:
    for param_combo_name, _ in param_combo_rmse.items():
        rmse_list = param_combo_rmse[param_combo_name]
        change_to_normal_list = param_combo_change_to_normal[param_combo_name]
        distance_to_goal_list = param_combo_distance_to_goal[param_combo_name]
        overall_results_file.write(f"{param_combo_name}:\n")

        overall_rmse = np.mean(rmse_list)
        std_rmse = np.std(rmse_list)
        overall_results_file.write(f"     Overall RMSE: {overall_rmse:.6f} ± {std_rmse:.6f}\n")
        for i in range(len(rmse_list)):
            overall_results_file.write(f"         {order_of_files[i]}: {rmse_list[i]:.6f}\n")

        overall_change_to_normal = np.mean(change_to_normal_list)
        std_change_to_normal = np.std(change_to_normal_list)
        overall_results_file.write(f"     Change to Normal RMSE: {overall_change_to_normal:.6f} ± {std_change_to_normal:.6f}\n")
      
        overall_distance_to_goal = np.mean(distance_to_goal_list)
        std_distance_to_goal = np.std(distance_to_goal_list)
        overall_results_file.write(f"     Distance to Goal: {overall_distance_to_goal:.6f} ± {std_distance_to_goal:.6f}\n")
        for i in range(len(distance_to_goal_list)):
            overall_results_file.write(f"         {order_of_files[i]}: {distance_to_goal_list[i]:.6f}\n")
        overall_results_file.write("\n\n")
        
        
        
        









