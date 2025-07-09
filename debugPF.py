import numpy as np
import matplotlib.pyplot as plt
import Models.functions as f
import Models.intentFunctions as iF
import Models.seqInference as seq
from dataFunctions import make_groundtruth, pretty_print_matrix, save_vector_arrays_txt, save_matrix_arrays_txt, save_state_comparison_txt, get_model_rmse, save_tracking_plot, save_variance_array_txt, save_particles_txt, save_vector_array_txt
from tqdm import tqdm

#groundtruth_filename = "Data/Generated/SE_track.txt"
debugging_folder = "Debugging/conv_iSE/Generated/simple_PF"
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
num_particles = 3
warmup_steps = 10

##Adding noise to create simulated sensor measurements if only groundtruth data available
sy = 0.5
noisy_data = [groundtruth[k] + np.random.normal(0, sy, 2) for k in range(Tmax)]


## Goal Creation and Convergence State##
G_prior = np.array([[10.0, 10.0]])
G_var = 10  #Unsure goal initially
Lamda_prior = np.array([[0.5, 0.5]])
Lambda_var = 10 #Unsure convergence rate initially

mk = [groundtruth[0, :] * np.ones([d, 2])]
mk[0][:-1] -= mk[0][-1]
mk[0] = np.vstack((mk[0], G_prior))
mk[0] = np.vstack((mk[0], Lamda_prior))

vk = [np.eye(d+2)]
vk[0][:-3, :-3] = f.iSE(t[1:], t[1:], s2/10, ls)
vk[0][-3, -3] = s2/10
vk[0][-2,-2] = G_var
vk[0][-1,-1] = Lambda_var

particles_1 = seq.sample_particles(mk[-1], vk[-1], num_particles)
particle_covariances_1 = np.array([vk[-1] for _ in range(num_particles)])

for k in range(10):
    new_particles_1, new_part_cov_1, all_F_conv_1, all_P_conv_1 = seq.propagate_particles(particles_1, particle_covariances_1, t, s2, ls, G_var, Lambda_var)

    y=noisy_data[1]
    used_sy = 100

    weights, meas_and_obv = seq.computeNorm_weights(new_particles_1, y, used_sy, k)

    particles_2 = seq.resample_particles(new_particles_1, weights)


save_matrix_arrays_txt(particles_1, new_particles_1, "particles_newParts.txt", "Particles", "Particles_post", debugging_folder)
save_vector_array_txt(meas_and_obv, "observations_and_expected_measurements.txt", "obs and meas", debugging_folder)
save_matrix_arrays_txt(new_particles_1, particles_2, "particles_newParts_resampled.txt", "Particles_post_propagate", "Particles_post_resample", debugging_folder)







