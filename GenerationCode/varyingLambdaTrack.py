import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from Models.functions import gen_iSE_track, gen_SE_track
from Models.intentFunctions import gen_goal_driven_track, gen_gp_bridge, gen_iSE_driven_track, gen_iSE_track_goal_converging
from dataFunctions import save_trajectory_plot, save_lambda_values_txt
import random
random.seed(24)
np.random.seed(24)

##################################
## Choices to Make ##
goal = np.array([50.0,50.0])
Tmax = 100
data_folder = "Data/Generated/giSE_convergingMeasModel/varyingLambda"
os.makedirs(data_folder, exist_ok=True)
trajectory_name = "1_varLambdaTrack"
trajectory_dir = os.path.join(data_folder, trajectory_name)
os.makedirs(trajectory_dir, exist_ok=True)

##################################
## Common Model Parameters ##
d = 5
s2 = 10
ls = 5

##################################
## Generate Lambda Values for the Time Sequence ##
## Using small random walk with increasing value ##
lambda_values = np.zeros(Tmax)
lambda_values[0] = 0.01
drift = 0.001
for k in range(1, Tmax):
    lambda_values[k] = lambda_values[k-1] + drift + np.random.normal(0, 0.005)
    lambda_values[k] = np.clip(lambda_values[k], 0.0001, 1)

##################################
## Generate Track ##
varying_lambda_track, _ , _, _, _= gen_iSE_track_goal_converging(Tmax, d, s2, ls, dim=2, dt=1, first_is_last=False, lambda_val=0.0, goal=goal, varying_lambda=True, lambda_values=lambda_values)
cur_header = f"d={d}, s2={s2}, l={ls}, goal={goal.tolist()}, drift={drift}"

##################################
## Save Track Plot and Lambda Values ##
save_trajectory_plot(varying_lambda_track, "varyingLam_track.png", trajectory_dir, true_goal=goal)
np.savetxt(os.path.join(trajectory_dir, "varyingLam_track.txt"), varying_lambda_track, header=cur_header)
save_lambda_values_txt(lambda_values, "lambda_values.txt", trajectory_dir)






