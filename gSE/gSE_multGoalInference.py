import numpy as np
import os
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
from dataFunctions import make_groundtruth, pretty_print_matrix, save_vector_arrays_txt, save_matrix_arrays_txt, save_state_comparison_txt, get_model_rmse, save_tracking_plot, extract_params_from_header, save_specifications_txt, save_variance_array_txt, steps_to_convergence
from tqdm import tqdm
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import random, re
import time
random.seed(24)
np.random.seed(24)


csg = True
gen_from_GP = True
separate_tracking = True
#Model Notes
UZH=False
quad_goal_options = [np.array([-20,20]), np.array([20,20]), np.array([20,-20]), np.array([-20,-20])]
quad_goal_indices = [0,1,2,3]
quad_goal_probs = [0.25, 0.25, 0.25, 0.25]
quad_sigma_g_values = [0.0, 0.1, 0.5, 1.0, 2.0]
quad_groundtruth_folder = "Data/Generated/Quad_BridgingGP/"
quad_debugging_folder_base = "Debugging/SE_MultTargets/Quad_PED/"

csg_goal_options = [np.array([50,100]), np.array([35,115]), np.array([65,115]), np.array([35, 90]), 
np.array([65, 90]), np.array([50, 85]), np.array([35, 65]), np.array([65, 65]),
np.array([50, 60]), np.array([35, 45]), np.array([65, 45])]
csg_goal_indices = [0,1,2,3,4,5,6,7,8,9,10]
csg_goal_probs = [0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05]
csg_sigma_g_values = [0.0, 0.1, 0.2]
csg_groundtruth_folder = "Data/Generated/CSG_Bridging/"
csg_debugging_folder_base = "Debugging/SE_MultTargets/CSG_PED/"

if csg:
    goal_options = csg_goal_options
    goal_indices = csg_goal_indices
    goal_probs = csg_goal_probs
    sigma_g_values = csg_sigma_g_values
    groundtruth_folder = csg_groundtruth_folder
    debugging_folder_base = csg_debugging_folder_base
else:
    goal_options = quad_goal_options
    goal_indices = quad_goal_indices
    goal_probs = quad_goal_probs
    sigma_g_values = quad_sigma_g_values
    groundtruth_folder = quad_groundtruth_folder
    debugging_folder_base = quad_debugging_folder_base



#Common parameters
s2 = 10
ls = 5 #Changed this for GP Generated Data, formerly 30
d = 5
dt = 1
sigma_y = 1.0
t = dt * np.arange(d,0,-1)
initialize_with_truth = True
G_var_values = [0, 1, 2, 5, 10]

goal_map = {tuple(goal): idx for idx, goal in enumerate(goal_options)}

#UNCOMMENT ONCE DEBUGGED
for subfolder in os.listdir(groundtruth_folder):
    print(subfolder)
    #Go through each of the quad groundtruths
    groundtruth_path = f"{groundtruth_folder}/{subfolder}/track.txt"
    groundtruth = make_groundtruth(groundtruth_path, UZH=UZH)
    Tmax = groundtruth.shape[0]
    debugging_folder = f"{debugging_folder_base}/{subfolder}"

    #If data is GP generated, won't end directly at the goal
    if gen_from_GP:
        goal_numbers = re.findall(r'\[(.*?)\]', subfolder)
        print(f"Goal numbers: {goal_numbers}")
        TRUE_GOAL = np.array([int(x) for x in goal_numbers[0].split()])
    else:
        TRUE_GOAL = groundtruth[-1,:]

    #IF S2 ARE DIFFERENT, GET THE CORRECT VALUE FROM FOLDER NAME
    if gen_from_GP:
        s2 = int(subfolder.split('_S2:')[1]) #s2/10 is used to initialize in final_double_swap then s2 itself is used in predict/update steps
    else:
        s2 = s2
    print(f"CHOSEN S2: {s2}")

    for sigma_g in sigma_g_values:
        for G_var in G_var_values:
            debugging_folder_sub = f"{debugging_folder}/SG:{sigma_g}/Gvar:{G_var}"

            #Track initialization
            if initialize_with_truth:
                init_point = groundtruth[0,:]
            else:
                init_point = np.array([50,50])

            ##Add noise to create simulated sensor measurements
            noisy_data = [groundtruth[k] + np.random.normal(0, sigma_y, 2) for k in range(Tmax)]
            ##Two samples drawn independently from N(0, sigma_y)

            #Initialize state mean and covariance for EACH goal option
            mk_goals = {}
            vk_goals = {}

            ################################
            # Posterior Weighted RMSE Calculation #
            ################################
            overall_rmse_total = 0.0
            overall_rmse_total_separate_states = 0.0
            ################################

            ################################
            # Including separate state group with higher sigma_g^2 and G_var for tracking purposes #
            ################################
            mk_goals_tracking = {}
            vk_goals_tracking = {}
            G_var_tracking = 10000
            sigma_g_tracking = 10
            ################################

            for i, goal_option in enumerate(goal_options):
                name = f"goal_{i}" 
                mk_goal = [init_point * np.ones([d, 2])]
                mk_goal[0] = np.vstack((mk_goal[0], goal_option))
                #print(f"mk_goal for goal option {goal_option}: {mk_goal}")

                vk_goal = [np.eye(d+1)]
                vk_goal[0][:-1, :-1] = f.SE(t,t,s2/10,ls)
                vk_goal[0][-1, -1] = G_var

                mk_goals[name] = mk_goal
                vk_goals[name] = vk_goal

                ################################
                # Tracking State Group #
                ################################
                if separate_tracking:
                    mk_goal_tracking = [init_point * np.ones([d, 2])]
                    mk_goal_tracking[0] = np.vstack((mk_goal_tracking[0], goal_option))

                    vk_goal_tracking = [np.eye(d+1)]
                    vk_goal_tracking[0][:-1, :-1] = f.SE(t,t,s2/10,ls)
                    vk_goal_tracking[0][-1, -1] = G_var_tracking

                    mk_goals_tracking[name] = mk_goal_tracking
                    vk_goals_tracking[name] = vk_goal_tracking
                ################################

            X_goals = {}
            S_goals = {}
            G_goals = {}
            S_goal_vars = {}
            running_likelihoods = {}

            ################################
            # Tracking State Group #
            ################################
            X_goals_tracking = {}
            S_goals_tracking = {}
            G_goals_tracking = {}
            S_goal_vars_tracking = {}

            X_From_Best_State = []
            ################################

            ##Unnecessary final objects - just for debugging##
            predicted_ys = {}
            log_likelihoods = {}
            posteriors_for_each_goal ={}
            x_residuals = {}
            y_residuals = {}
            log_norm_consts = {}
            log_of_exponent_xs = {}
            log_of_exponent_ys = {}
            PEDs = {}
            predicted_states_used = {}
            predicted_covars_used = {}
            ##END OF DEBUG OBJECT##

            for i, goal_option in enumerate(goal_options):
                name = f"goal_{i}"
                X_goal = np.zeros([Tmax,2])
                S_goal = np.zeros([Tmax]) #Uncertainty at each time step for goal model
                G_goal = np.zeros([Tmax,2])
                S_goal_var = np.zeros([Tmax])
                running_likelihoods[name] = np.zeros([Tmax])

                X_goals[name] = X_goal
                S_goals[name] = S_goal
                G_goals[name] = G_goal
                S_goal_vars[name] = S_goal_var
                running_likelihoods[name][0] = 0.0 #Initialize step in Algorithm 2 of Bridging Paper

                ################################
                # Tracking State Group #
                ################################
                if separate_tracking:
                    X_goal_tracking = np.zeros([Tmax,2])
                    S_goal_tracking = np.zeros([Tmax])
                    G_goal_tracking = np.zeros([Tmax,2])
                    S_goal_var_tracking = np.zeros([Tmax])

                    X_goals_tracking[name] = X_goal_tracking
                    S_goals_tracking[name] = S_goal_tracking
                    G_goals_tracking[name] = G_goal_tracking
                    S_goal_vars_tracking[name] = S_goal_var_tracking
                ################################

                ##Unnecessary Debug Objects##
                predicted_ys[name] = np.zeros([Tmax,2])
                log_likelihoods[name] = np.zeros([Tmax])
                posteriors_for_each_goal[name] = np.zeros([Tmax])
                x_residuals[name] = np.zeros([Tmax])
                y_residuals[name] = np.zeros([Tmax])
                log_norm_consts[name] = np.zeros([Tmax])
                log_of_exponent_xs[name] = np.zeros([Tmax])
                log_of_exponent_ys[name] = np.zeros([Tmax])
                PEDs[name] = np.zeros([Tmax])
                predicted_states_used[name] = [None] * Tmax
                predicted_covars_used[name] = [None] * Tmax
                ##END OF DEBUG OBJECT##

            likelihoods = {}
            for i, goal_option in enumerate(goal_options):
                name = f"goal_{i}"
                likelihoods[name] = np.zeros([Tmax])

            best_goal = np.zeros([Tmax,2])
            all_likelihoods = np.zeros([Tmax, len(goal_options)])

            goal_estimates_all = np.zeros([len(goal_options), Tmax, 2])
            goal_vars_all = np.zeros([len(goal_options), Tmax, 1])
            goal_indices_all = np.zeros([len(goal_options), Tmax])

            # Initialize with a recognizable value to see if updates are happening
            goal_estimates_all.fill(-999)  # Use -999 as a marker for uninitialized values

            ##Actual Tracking Portion##
            #For each time step, complete Kalman Filter for each goal-priored state##
            start_time_before_tracking = time.time()
            for k in range(Tmax): ##CHANGE THIS BACK TO TMAX ##
                for model_idx, goal_option in enumerate(goal_options):
                    name = f"goal_{model_idx}"
                    mk_current = mk_goals[name][-1]
                    vk_current = vk_goals[name][-1]

                    m_pred, v_pred, F_goal, P_goal = iF.g_se_pred(t,mk_current,vk_current,s2,ls, sigma_g)

                    ################################
                    # Tracking State Group #
                    ################################
                    if separate_tracking:
                        mk_current_tracking = mk_goals_tracking[name][-1]
                        vk_current_tracking = vk_goals_tracking[name][-1]
                        m_pred_tracking, v_pred_tracking, F_goal_tracking, P_goal_tracking = iF.g_se_pred(t,mk_current,vk_current,s2,ls, sigma_g_tracking)
                    ################################

                    y = noisy_data[k]
                    datum = y

                    #TRYING PED ROUTE##
                    #Get predicted observation
                    H = np.zeros((1, m_pred.shape[0]))
                    H[0,0] = 1
                    H[0,-1] = 1
                    y_pred = H @ m_pred
                    predicted_states_used[name][k] = m_pred.copy()
                    y_pred_corrected = y_pred.flatten()
                    predicted_ys[name][k,:] = y_pred
                    
                    ##TRYING MULTIVAR NORMAL WAY##
                    #Get Covariance Matrix
                    #Observation noise matrix
                    sigma_y_for_PED = sigma_y
                    R = np.eye(2) * sigma_y_for_PED**2
                    S_k = H @ v_pred @ H.T + R 
                    predicted_covars_used[name][k] = S_k.copy()
                    
                    goal_option_dist = multivariate_normal(mean=y_pred_corrected, cov=S_k)
                    #PED = goal_option_dist.pdf(datum)
                    PED = goal_option_dist.logpdf(datum)
                    PEDs[name][k] = PED
                    
                    #Update the running log portion by using the log of the product of the likelihood and prior
                    if k == 0:
                        #running_likelihoods[name][k] = np.exp(log_likelihood)
                        running_likelihoods[name][k] = 0.0 #Going to skip the first step because there hasn't been an update yet, set to prior
                    else:
                        #running_likelihoods[name][k] = np.exp(log_likelihood) * running_likelihoods[name][k-1] #Like "Update likelihood" in Bridging Paper in Algorithm 2
                        running_likelihoods[name][k] = PED + running_likelihoods[name][k-1]
                    ##END OF PED ROUTE##

                    m_up, v_up, KGN, y_in= iF.g_update(datum, m_pred, v_pred, sigma_y)

                    mk_goals[name].append(m_up)
                    vk_goals[name].append(v_up)

                    ################################
                    # Tracking State Group #
                    ################################
                    if separate_tracking:
                        m_up_tracking, v_up_tracking, KGN_tracking, y_in_tracking = iF.g_update(datum, m_pred_tracking, v_pred_tracking, sigma_y)
                        mk_goals_tracking[name].append(m_up_tracking)
                        vk_goals_tracking[name].append(v_up_tracking)
                        X_goals_tracking[name][k,:] = m_up_tracking[0,:] + m_up_tracking[-1,:]
                        S_goals_tracking[name][k] = v_up_tracking[0,0] + v_up_tracking[-1,-1]
                        G_goals_tracking[name][k,:] = m_up_tracking[-1,:]
                        S_goal_vars_tracking[name][k] = v_up_tracking[-1,-1]
                    #############################################

                    X_goals[name][k,:] = m_up[0,:] + m_up[-1,:]
                    S_goals[name][k] = v_up[0,0] + v_up[-1,-1]
                    G_goals[name][k,:] = m_up[-1,:]
                    S_goal_vars[name][k] = v_up[-1,-1]

                    cur_pred_loc = m_up[0,:] + m_up[-1,:]
                    cur_pred_cov = v_up[0,0] + v_up[-1,-1]

                #End of for loop for each goal destination
                #Calculate the final posterior for each goal destination at this time step (u in Bridging Paper)    
                goal_posteriors = np.zeros([len(goal_options)])
                

                #Unnormalized posteriors
                unnorm_posteriors = np.zeros([len(goal_options)])
                for i, goal_option in enumerate(goal_options):
                    name = f"goal_{i}"
                    unnorm_posteriors[i] = running_likelihoods[name][k] + np.log(goal_probs[i])
                denominator = logsumexp(unnorm_posteriors)
                
                for i, goal_option in enumerate(goal_options):
                    name = f"goal_{i}"
                    #Since running likelihoods are in negative log space, exponentiate before multiplying by priors
                    # numerator = running_likelihoods[name][k] + np.log(goal_probs[i])
                    # denominator = np.sum([np.exp(running_likelihoods[f"goal_{j}"]) * goal_probs[j] for j in range(len(goal_options))])
                    # posterior = numerator / denominator
                    posterior = np.exp(unnorm_posteriors[i] - denominator)
                    goal_posteriors[i] = posterior
                    posteriors_for_each_goal[name][k] = posterior
                
                #Choose goal with highest posteriorl
                best_goal_index = np.argmax(goal_posteriors)
                goal_estimates_all[:,k,:] = goal_options[best_goal_index] #This will make all the models, regardless of initialization, have the same goal estimate
                goal_indices_all[:,k] = best_goal_index

                ################################
                #Posterior Weighted RMSE Calculation#
                # Debug: Check posteriors sum to 1
                posterior_sum = np.sum(goal_posteriors)
                if abs(posterior_sum - 1.0) > 1e-6:
                    print(f"Warning: Posteriors don't sum to 1 at time {k}, sum = {posterior_sum}")
                # Sum across goal models, not across spatial dimensions
                overall_pred_loc = np.zeros(2)
                for i in range(len(goal_options)):
                    overall_pred_loc += goal_posteriors[i] * X_goals[f"goal_{i}"][k,:]  
                current_difference_squared = (np.linalg.norm(overall_pred_loc - groundtruth[k,:]))**2             
                overall_rmse_total += current_difference_squared
                ################################

                ################################
                # Choose the predicted location from the state group with the highest posterior
                if separate_tracking:
                    overall_pred_loc_sep_states = np.zeros(2)
                    for i in range(len(goal_options)):
                        overall_pred_loc_sep_states += goal_posteriors[i] * X_goals_tracking[f"goal_{i}"][k,:]
                    current_difference_squared_sep_states = (np.linalg.norm(overall_pred_loc_sep_states - groundtruth[k,:]))**2
                    overall_rmse_total_separate_states += current_difference_squared_sep_states
                ################################
                    
            end_time_after_tracking = time.time()
            #Debugging Files ##
            all_rmse = {}
            for i, goal_option in enumerate(goal_options):
                name = f"goal_{i}"
                X_goal = X_goals[name]
                G_goal = G_goals[name]
                # Create list of false goals (all goals except TRUE_GOAL)
                false_goals = [goal for goal in goal_options if not np.array_equal(goal, TRUE_GOAL)]
                debugging_folder_plots = os.path.join(debugging_folder_sub, "Plots")
                if not os.path.exists(debugging_folder_plots):
                    os.makedirs(debugging_folder_plots)
                if csg:
                    save_tracking_plot(groundtruth, noisy_data, X_goal, G_goal[15:], "g-SE", f"Goal_{goal_options[i]}.png", debugging_folder_plots, show_Target=True, true_goal=TRUE_GOAL, false_goals=false_goals)
                else:
                    save_tracking_plot(groundtruth, noisy_data, X_goal, G_goal, "g-SE", f"Goal_{goal_options[i]}.png", debugging_folder_plots, show_Target=True, true_goal=TRUE_GOAL, false_goals=false_goals)
                cur_rmse = get_model_rmse(X_goal, groundtruth)
                all_rmse[name] = cur_rmse

            ################################
            #Best Prediction RMSE#
            if separate_tracking:
                overall_rmse_under_sqrt_sep_states = overall_rmse_total_separate_states / Tmax
                overall_rmse_sep_states = np.sqrt(overall_rmse_under_sqrt_sep_states)
            ################################

            ################################
            # Posterior Weighted RMSE Calculation #
            ################################
            overall_rmse_under_sqrt = overall_rmse_total / Tmax
            overall_rmse = np.sqrt(overall_rmse_under_sqrt)
            ################################

            # Save RMSE values to text file
            with open(os.path.join(debugging_folder_sub, "rmse_results.txt"), 'w') as file_handle:
                file_handle.write("RMSE Values by Model\n")
                file_handle.write("=" * 30 + "\n\n")
                for model_name, rmse_value in all_rmse.items():
                    file_handle.write(f"{model_name}: {rmse_value:.6f}\n")
                mean_rmse = np.mean(list(all_rmse.values()))
                file_handle.write(f"\nMean RMSE: {mean_rmse:.6f}\n")
                file_handle.write(f"Std RMSE: {np.std(list(all_rmse.values())):.6f}\n")
                file_handle.write(f"Posterior Weighted RMSE: {overall_rmse:.6f}\n")
                if separate_tracking:
                    file_handle.write(f"Best Overall Prediction RMSE: {overall_rmse_sep_states:.6f}\n")
                file_handle.write(f"Time taken for tracking: {end_time_after_tracking - start_time_before_tracking:.6f} seconds\n")
                
                # Calculate average end distance
                dists = 0.0
                for i, goal_option in enumerate(goal_options):
                    end_dist = np.linalg.norm(G_goals[f'goal_{i}'][-1,:] - TRUE_GOAL)
                    file_handle.write(f"End Goal for model {i}: {G_goals[f'goal_{i}'][-1,:]}\n")
                    dists += end_dist
                avg_end_distance = dists/len(goal_options)
                file_handle.write(f"Average End Distance: {avg_end_distance:.6f}\n\n")

                #Calculate steps to convergence for each goal option
                for i, goal_option in enumerate(goal_options):
                    if i > 0:
                        break
                    true_x = int(TRUE_GOAL[0])
                    true_y = int(TRUE_GOAL[1])
                    TRUE_GOAL_TUPLE = (true_x, true_y)
                    cur_steps_to_convergence, percentage_correct = steps_to_convergence(goal_indices_all[i,:], goal_map[TRUE_GOAL_TUPLE])
                    file_handle.write(f"Steps to convergence for model with sigma_g: {sigma_g} and G_var: {G_var}: {cur_steps_to_convergence}\n")
                    file_handle.write(f"Percentage of time in correct goal state: {percentage_correct:.2f}%\n")

            for i in range(1):
                np.savetxt(os.path.join(debugging_folder_sub, f"predicted_goals.txt"), goal_estimates_all[i, :, :], fmt='%d')
            for i in range(1):
                np.savetxt(os.path.join(debugging_folder_sub, f"variances_goals.txt"), S_goal_vars[f"goal_{i}"], fmt='%.6f')

            # Save posteriors to separate file
            with open(os.path.join(debugging_folder_sub, "posteriors.txt"), 'w') as posteriors_file:
                posteriors_file.write("Posterior Probabilities for Each Goal at Each Time Step\n")
                posteriors_file.write("=" * 60 + "\n\n")
                
                for k in range(Tmax):
                    posteriors_file.write(f"Time Step = {k}\n")
                    for i, goal_option in enumerate(goal_options):
                        name = f"goal_{i}"
                        posterior = posteriors_for_each_goal[name][k]
                        posteriors_file.write(f"  {goal_option}: {posterior:.6f}\n")
                    posteriors_file.write("\n\n")

            # Save likelihood debugging information
            with open(os.path.join(debugging_folder_sub, "likelihood_debug.txt"), 'w') as debug_file:
                debug_file.write("Likelihood Debug Information\n")
                debug_file.write("=" * 40 + "\n\n")
                debug_file.write(f"Goal options: {goal_options}\n")
                debug_file.write(f"Goal probabilities: {goal_probs}\n\n")
                
                # Save likelihoods for first few time steps
                for k in range(min(25, Tmax)):
                    debug_file.write(f"Time step {k}:\n")
                    for model_idx in range(len(goal_options)):
                        debug_file.write(f"  Model {model_idx} (initialized with {goal_options[model_idx]}):\n")
                        debug_file.write(f"    Predicted state (m_pred):\n")
                        for i, state_component in enumerate(predicted_states_used[f'goal_{model_idx}'][k]):
                            debug_file.write(f"      Component {i}: {state_component}\n")
                        debug_file.write(f"    Predicted covariance (v_pred):\n")
                        pretty_print_matrix(predicted_covars_used[f'goal_{model_idx}'][k], "v_pred", debug_file)
                        debug_file.write(f"    Predicted goal: {G_goals[f'goal_{model_idx}'][k,:]}\n")
                        debug_file.write(f"    Goal variance: {S_goal_vars[f'goal_{model_idx}'][k]:.6f}\n")
                        debug_file.write(f"    Predicted y: {predicted_ys[f'goal_{model_idx}'][k,:]}\n")
                        debug_file.write(f"    Data: {noisy_data[k]}\n")
                        debug_file.write(f"    PED: {PEDs[f'goal_{model_idx}'][k]}\n")
                        debug_file.write(f"    Running likelihood: {running_likelihoods[f'goal_{model_idx}'][k]}\n")
                        debug_file.write(f"    Posterior: {posteriors_for_each_goal[f'goal_{model_idx}'][k]}\n")
                        debug_file.write(f"    Selected goal: {goal_estimates_all[model_idx, k, :]}\n")
                    debug_file.write("\n")










