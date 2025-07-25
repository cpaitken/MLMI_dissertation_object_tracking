## Basic Particle Filter Implementation ##

## Currently just used to make sure the PF works and has been tested
## on a few trajectories with fixed but unknown lambda to see if 
## performance is okay

######################################
## Imports ##
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import Models.functions as f
import Models.intentFunctions as iF
import Models.seqInference as seq
from dataFunctions import make_groundtruth, get_model_rmse, save_tracking_plot, save_vector_array_txt
import random, copy
random.seed(24)
np.random.seed(24)


######################################
## Getting data file ##
UZH = True
groundtruth_file = "Data/UZH/Easy/UZH_1.txt"
groundtruth = make_groundtruth(groundtruth_file, UZH=UZH)
if UZH:
    groundtruth = groundtruth[::20]
debugging_folder = "Debugging/conv_iSE/varyingLambda/initial_debug_UZH"
T = groundtruth.shape[0]
true_goal = np.array([0.86, -18])
true_lambda = 0.05

######################################
## Common Model Parameters/Choice to Make ##
initialize_goal_with_truth = False
initialize_lambda_with_truth = True
num_particles = 100

dt = 1
d =5
Tmax = T 
s2=10
ls=5
sigma_g = 10
t = dt * np.arange(d, 0, -1)



######################################
## Making noisy data ##
sy = 0.25
noisy_data = [groundtruth[k] + np.random.normal(0, sy, 2) for k in range(Tmax)]

######################################
## Initializations ##
if initialize_goal_with_truth:
    G_prior = true_goal
else:
    G_prior = np.array([0,0])
G_var = 100 #sure goal initially

if initialize_lambda_with_truth:
    lambda_prior = true_lambda
else:
    lambda_prior = 0.05
lambda_var = 0.05
lambda_arr = np.array([lambda_prior, lambda_prior])

## Normal iSE ##
mk_normal= [groundtruth[0, :] * np.ones([d, 2])]
mk_normal[0][:-1] -= mk_normal[0][-1]  # iSE-1 style offset
vk_normal = [np.eye(d)]
vk_normal[0][:-1,:-1] = f.iSE(t[1:],t[1:],s2/10,ls)
vk_normal[0][-1,-1] = s2/10
# for row in vk_normal[0]:
#     print("  " + "  ".join(f"{val:8.4f}" for val in row) + "\n")

## Goal-iSE ##
mk_goal = [groundtruth[0, :] * np.ones([d, 2])]
mk_goal[0][:-1] -= mk_goal[0][-1]
mk_goal[0] = np.vstack((mk_goal[0], G_prior))
#Include lambda in the state when initializing
#mk_goal[0] = np.vstack((mk_goal[0], lambda_arr))


P_goal = np.zeros((d+1, d+1))
P_goal[:d-1, :d-1] = f.iSE(t[1:], t[1:], s2/10, ls)
P_goal[-2, -2] = s2/10
P_goal[-1,-1] = G_var
#P_goal[-1,-1] = lambda_var

vk_goal = [P_goal]
# for row in vk_goal[0]:
#     print("  " + "  ".join(f"{val:8.4f}" for val in row) + "\n")

######################################
## Final Objects ##
X_normal = np.zeros([Tmax,2]) 
S_normal = np.zeros([Tmax])

X_goal = np.zeros([Tmax,2]) 
S_goal = np.zeros([Tmax])
G_goal = np.zeros([Tmax, 2]) 
S_goal_var = np.zeros([Tmax])
Lambda_goal = np.zeros([Tmax])

######################################
## Debugging Objects ##
normal_F_aug = []
goal_F_aug = []
normal_Covar = []
goal_Covar = []

normal_predicted_means = []
goal_predicted_means = []
normal_updated_means = []
goal_updated_means = []
goal_predicted_lambdas = []
goal_predicted_goals = []

######################################
## Tracking Loop - Normal iSE ##




######################################
## Tracking Loop - Goal-iSE ##
## Sample initial lambda particles ##
particle_dict = {}
initial_lambda_particles = np.random.normal(lambda_prior, np.sqrt(lambda_var), num_particles)
#print("Initial lambda particles are:", initial_lambda_particles) #DEBUG
#Set up different states for each particle
for i in range(num_particles):
    particle = initial_lambda_particles[i]
    particle_mk_goal = mk_goal[0].copy()
    #Set to new lambda value
    particle_dict[i] = {
        "lambda": particle,
        "mk": particle_mk_goal,
        "vk": vk_goal[0].copy(),
        "incremental_weight": 1/num_particles, 
        "normalized_weight": 1/num_particles
    }
# for key, value in particle_dict.items(): #DEBUG
#     print(key, particle_dict[key]["mk"])

#Start for loop for each time step
for k in range(Tmax):
    #Get observation
    y = noisy_data[k]

    #If not the first time step, update the lambda
    # if k > 0:
    #     for key, value in particle_dict.items():
    #         new_lambda = np.random.normal(value["lambda"], np.sqrt(lambda_var))
    #         new_lambda = np.clip(new_lambda, 0.0001, 1)
    #         value["lambda"] = new_lambda

    #Start for loop for each lambda
    current_inc_weights = []
    for key, value in particle_dict.items():
         #Predict step with saved lambda
         part_m_pred, part_v_pred, part_F_goal, part_P_goal = iF.converging_ise_pred(t+dt*(k+1), value["mk"], value["vk"], s2, ls, sigma_g=sigma_g)
         #print("Particle", key, "predicted mean is:", part_m_pred) #DEBUG
         #Update step with predicted state, cov, observation, lambda - return observation likelihood
         part_m_up, part_v_up, part_KG, part_decay_rate, part_obs_likelihood = iF.fixed_lambda_kf_update_for_PF(y, part_m_pred, part_v_pred, sy, k, value["lambda"])
         #Calculate individual incremental importance weight as the observation likelihood
         value["incremental_weight"] = part_obs_likelihood
         value["mk"] = part_m_up
         value["vk"] = part_v_up
         current_inc_weights.append(value["incremental_weight"])

    #END OF LOOP FOR EACH PARTICLE
    #print("Current ground truth is:", groundtruth[k])
    #Normalize weights
    weight_denominator = logsumexp(current_inc_weights)
    for key, value in particle_dict.items():
        value["normalized_weight"] = np.exp(value["incremental_weight"] - weight_denominator)

    
    #Predicted location is weighted sum of predicted locations for the particles
    overall_pred_loc = np.zeros(2)
    for key, value in particle_dict.items():
        # print("Shape of mk is:", value["mk"].shape)
        # print("Mk is:", value["mk"])
        current_pred_loc = iF.conv_ise_measure(value["mk"], value["lambda"], k)
        # print("Current predicted location is:", current_pred_loc)
        # print("Shape of current predicted location is:", current_pred_loc.shape)
        # print("Normalized weight is:", value["normalized_weight"])
        # print("Shape of normalized weight is:", value["normalized_weight"].shape)
        overall_pred_loc += value["normalized_weight"]*current_pred_loc[0]
    X_goal[k,:] = overall_pred_loc
    #print("Overall predicted location at time", k, "is:", overall_pred_loc)


    #Predicted goal is weighted sum of predicted goals for the particles
    overall_pred_goal = np.zeros(2)
    for key, value in particle_dict.items():
        overall_pred_goal += value["normalized_weight"]*value["mk"][-1]
    G_goal[k,:] = overall_pred_goal
    #print("Overall predicted goal at time", k, "is:", overall_pred_goal)


    #Predicted lambda is weighted sum of predicted lambdas for the particles
    #Keep best lambda for debugging
    overall_pred_lambda = np.zeros(1)
    for key, value in particle_dict.items():
        overall_pred_lambda += value["normalized_weight"]*value["lambda"]
    Lambda_goal[k] = overall_pred_lambda[0]
    #print("Overall predicted lambda at time", k, "is:", Lambda_goal[k])

    
    #Resample the particles and assign new weights to 1/N (essentially just means equally likely to be chosen for next round)
    #Using Systematic Resampling (Page 13 of Tutorial)
    normalized_weights = np.array([value["normalized_weight"] for value in particle_dict.values()])
    resampled_indices = iF.systematic_resample_particles(normalized_weights)
    resampled_particle_dict = {}
    for i in range(num_particles):
        #print("Resampled index is:", resampled_indices[i])
        particle_to_continue = particle_dict[resampled_indices[i]]
        #Use deepcopy so in subsequent runs the kalman filter runs separately
        resampled_particle_dict[i] = copy.deepcopy(particle_to_continue)
    particle_dict = resampled_particle_dict
    # for key, value in particle_dict.items():
    #     print("Particle", key, "has mk:", value["mk"])
    print("did step", k)

######################################
## Save results and debugging ##
overall_rmse = get_model_rmse(X_goal, groundtruth)
os.makedirs(debugging_folder, exist_ok=True)
with open(os.path.join(debugging_folder, "results.txt"), 'w') as results_file:
    results_file.write("RMSE: " + str(overall_rmse) + "\n")
    for i in range(Tmax):
        results_file.write("Lambda at time " + str(i) + " is: " + str(Lambda_goal[i]) + "\n")

save_vector_array_txt(X_goal, "predicted_loc.txt", "X_goal", debugging_folder)
save_vector_array_txt(G_goal, "predicted_goal.txt", "G_goal", debugging_folder)
save_tracking_plot(groundtruth, noisy_data, X_goal, G_goal, "giSE-VaryingLambda", "giSE_varyingLambda_debug.png", debugging_folder, show_Target=True, true_goal=true_goal)