import numpy as np
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
import Models.seqInference as seq
from dataFunctions import make_groundtruth, pretty_print_matrix, save_vector_arrays_txt, save_matrix_arrays_txt, save_state_comparison_txt, get_model_rmse, save_tracking_plot, save_variance_array_txt, save_particles_txt, save_vector_array_txt
from tqdm import tqdm


#groundtruth_filename = "Data/Generated/SE_track.txt"
debugging_folder = "Debugging/conv_iSE/Generated/goal_iSE_track"
UZH = False
#groundtruth = groundtruth[::20]
groundtruth_filename = "Data/Generated/goal_iSE_track.txt"
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
num_particles = 500
warmup_steps = 10



##Adding noise to create simulated sensor measurements if only groundtruth data available
sy = 1.0
noisy_data = [groundtruth[k] + np.random.normal(0, sy, 2) for k in range(Tmax)]


## Goal Creation and Convergence State##
G_prior = np.array([[0.0, 0.0]])
G_var = 10  #Unsure goal initially
Lamda_prior = np.array([[0.0, 0.0]])
Lambda_var = 10 #Unsure convergence rate initially

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
mk[0] = np.vstack((mk[0], Lamda_prior))

vk = [np.eye(d+2)]
vk[0][:-3, :-3] = f.iSE(t[1:], t[1:], s2/10, ls)
vk[0][-3, -3] = s2/10
vk[0][-2,-2] = G_var
vk[0][-1,-1] = Lambda_var

#Storage for full debugging vectors
normal_F_aug = []
goal_F_aug = []
goal_P_conv = []
normal_P_conv = []

normal_predicted_means = []
goal_predicted_means = []
normal_updated_means = []
goal_updated_means = []

#Debugging particles
particles_post_propagate = []
observations_and_expected_measurements = []
resampled_particles = []
estimated_states = []
all_weights = []



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
    m_predN, v_predN, F_aug, P = f.ise1_pred(t, mkN[-1], vkN[-1], s2, 1.0)
    

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
    normal_P_conv.append(P.copy())
    normal_predicted_means.append(m_predN.copy())
    normal_updated_means.append(m_upN.copy())
#else:
#Particle Filtering Portion and Initialization
particles = seq.sample_particles(mk[-1], vk[-1], num_particles)
particle_covariances = np.array([vk[-1] for _ in range(num_particles)])
weights = np.ones(num_particles) / num_particles

#print("Initialized particles:", particles)
for k in range(Tmax):
    #Predict using the goal convering iSE model and particle filtering
    new_particles, new_part_cov, all_F_conv, all_P_conv = seq.propagate_particles(particles, particle_covariances, t, s2, ls, G_var, Lambda_var)
    
    #Noisy Observation
    y = noisy_data[k]
    if k < warmup_steps:
        used_sy = 100
    else:
        used_sy = sy

    #Compute weights given the observation
    weights, meas_and_obv = seq.computeNorm_weights(new_particles, y, used_sy, k)
    
    #Resample particles given the weights
    particles = seq.resample_particles(new_particles, weights)
    
    particle_covariances = new_part_cov

    #Estimate state at this time step
    state_estimate = np.average(particles, axis=0, weights=weights)
    
    current_predicted_y = iF.conv_ise_measure(state_estimate, sy, k)
    X_goal[k, :] = current_predicted_y
    G_goal[k, :] = state_estimate[-2, :]

    ##DEBUGGING LISTS##
    observations_and_expected_measurements.append(meas_and_obv)
    all_weights.append(weights.copy())
    resampled_particles.append(particles.copy())
    estimated_states.append(state_estimate.copy())
    particles_post_propagate.append(new_particles.copy())
    goal_F_aug.append(all_F_conv[-1].copy())
    goal_P_conv.append(all_P_conv[-1].copy())

save_tracking_plot(groundtruth, noisy_data, X_goal, G_goal, X_normal, "Goal-iSE", "iSE", "ComparisonPlot.png", debugging_folder)
save_matrix_arrays_txt(normal_F_aug[:5], goal_F_aug[:5], "transitionMatrices.txt", "F_aug", "F_goal", debugging_folder)
save_matrix_arrays_txt(normal_P_conv[:5], goal_P_conv[:5], "covarianceMatrices.txt", "P", "P_goal", debugging_folder)
save_particles_txt(particles_post_propagate, "particles_post_propagate.txt", debugging_folder)
save_particles_txt(resampled_particles, "resampled_particles.txt", debugging_folder)
save_particles_txt(estimated_states, "estimated_states.txt", debugging_folder)
save_vector_array_txt(observations_and_expected_measurements, "observations_and_expected_measurements.txt", "obs and meas", debugging_folder)
save_vector_array_txt(all_weights, "weights.txt", "weights", debugging_folder)
# save_vector_arrays_txt(normal_predicted_means, goal_predicted_means, "predictedMeans.txt", "m_predN", "m_pred", debugging_folder)
# save_vector_arrays_txt(normal_updated_means, goal_updated_means, "updatedMeans.txt", "m_updN", "m_upd", debugging_folder)
# save_variance_array_txt(S_goal_var, "goalVariances.txt", debugging_folder)
# save_variance_array_txt(S_normal, "normalLocationVariances.txt", debugging_folder)
# save_variance_array_txt(S_goal, "goalLocationVariances.txt", debugging_folder)

se_rmse = get_model_rmse(X_normal, groundtruth)
gse_rmse = get_model_rmse(X_goal, groundtruth)
print(f"RMSE of SE model:  {get_model_rmse(X_normal, groundtruth):.4f}")
print(f"RMSE of goal model:  {get_model_rmse(X_goal, groundtruth):.4f}")

print("Saved debugging vectors and matrices to folder Debugging")



