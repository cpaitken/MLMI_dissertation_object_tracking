import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from Models.functions import gen_iSE_track, gen_SE_track
from Models.intentFunctions import gen_goal_driven_track, gen_gp_bridge, gen_iSE_driven_track, gen_iSE_track_goal_converging
from dataFunctions import save_trajectory_plot
import random
random.seed(24)
np.random.seed(24)
#Parameter settings

d= 5
s2 = 10 #First s2 mentioned in paper
ls = 5 #Also mentioned in paper


goal = np.array([50.0,50.0])
#lambda_vals = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
#Tmax_vals = [50, 100, 250, 500]
#lambda_val = 0.05
Tmax = 100

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data/Generated'))
os.makedirs(data_dir, exist_ok=True)
trajectory_name = "giSE_convergingMeasModel/DEBUGVARLAMBDA"
trajectory_dir = os.path.join(data_dir, trajectory_name)
os.makedirs(trajectory_dir, exist_ok=True)

# fred_dir = os.path.join(data_dir, "Fred's_iSE")
# os.makedirs(fred_dir, exist_ok=True)
# Tmax_iSE = 500
# iSE_track = gen_iSE_track(Tmax_iSE, d,s2,ls)
# np.savetxt(os.path.join(fred_dir, "iSE_track.txt"), iSE_track)
# save_trajectory_plot(iSE_track, "iSE_track.png", fred_dir)

#SE_track = gen_SE_track(Tmax, d,s2,ls)

#goal_SE_track = gen_goal_driven_track(Tmax,d,s2,ls,goal)
#header_driven_track = f"d={d}, s2={s2}, l={ls}, Tmax={Tmax}, goal={goal.tolist()}"

#goalConditionedGP_track = gen_gp_bridge(Tmax, s2, ls, goal)

# iSE_goal_driven_track = gen_iSE_driven_track(Tmax, d, s2, ls, goal)
# header_ise_track = f"d={d}, s2={s2}, l={ls}, goal={goal.tolist()}"

#SECTION FOR TESTING TMAX AND LAMBDA VALS##
# for Tmax in Tmax_vals:  
#     Tmax_dir = os.path.join(trajectory_dir, f"Tmax_{Tmax}")
#     os.makedirs(Tmax_dir, exist_ok=True)
#     for lambda_val in lambda_vals:
#         lambda_dir = os.path.join(Tmax_dir, f"lambda_{lambda_val}")
#         os.makedirs(lambda_dir, exist_ok=True)
#         iSE_track_goalConv_measModel, iSE_contributions, goal_contributions, decay_rate, ise_contributions_final = gen_iSE_track_goal_converging(Tmax, d, s2, ls, dim=2, dt=1, first_is_last=False, lambda_val=lambda_val, goal=goal)
#         header_ise_track = f"d={d}, s2={s2}, l={ls}, goal={goal.tolist()}, lambda_val={lambda_val}"
#         file_name = f"Tmax:{Tmax}_lambda:{lambda_val}.txt"
#         np.savetxt(os.path.join(lambda_dir, file_name), iSE_track_goalConv_measModel, header=header_ise_track)
#         save_trajectory_plot(iSE_track_goalConv_measModel, file_name.replace(".txt", ".png"), lambda_dir)

##SECTION FOR MAKING MULTIPLE TRACKS OF TMAX=250 and lambda=0.01##
lambda_vals = [0.05]
for lambda_val in lambda_vals:
    lambda_folder = os.path.join(trajectory_dir, f"lambda_{lambda_val}")
    os.makedirs(lambda_folder, exist_ok=True)
    for i in range(10):
        iSE_track_goalConv_measModel, iSE_contributions, goal_contributions, decay_rate, ise_contributions_final = gen_iSE_track_goal_converging(Tmax, d, s2, ls, dim=2, dt=1, first_is_last=False, lambda_val=lambda_val, goal=goal)
        header_ise_track = f"d={d}, s2={s2}, l={ls}, goal={goal.tolist()}, lambda_val={lambda_val}"
        file_name = f"Tmax:{Tmax}_lambda:{lambda_val}_{i}.txt"
        np.savetxt(os.path.join(lambda_folder, file_name), iSE_track_goalConv_measModel, header=header_ise_track)
        save_trajectory_plot(iSE_track_goalConv_measModel, file_name.replace(".txt", ".png"), lambda_folder, true_goal=goal)



# np.savetxt(os.path.join(data_dir, "iSE_track.txt"), iSE_track)
# save_trajectory_plot(iSE_track, "iSE_track.png", trajectory_dir)
#np.savetxt(os.path.join(data_dir, "SE_track.txt"), SE_track)
#np.savetxt(os.path.join(trajectory_dir, "lowerS2_goal_SE_track.txt"), goal_SE_track, header=header_driven_track)
#save_trajectory_plot(goal_SE_track, "lowerS2_goal_SE_track.png", trajectory_dir)
#np.savetxt(os.path.join(data_dir, "goalMeanGP_track.txt"), goalMeanGP_track)
#np.savetxt(os.path.join(data_dir, "goalConditionedGP_track.txt"), goalConditionedGP_track)
#np.savetxt(os.path.join(trajectory_dir, "goal_iSE_track.txt"), iSE_goal_driven_track, header=header_ise_track)
#save_trajectory_plot(iSE_goal_driven_track, "ise_goal_track.png", trajectory_dir)



# np.savetxt(os.path.join(trajectory_dir, "goal_iSE_track_newMeasModel.txt"), iSE_track_goalConv_measModel, header=header_ise_track)
# save_trajectory_plot(iSE_track_goalConv_measModel, "goal_iSE_track_newMeasModel.png", trajectory_dir)
# np.savetxt(os.path.join(trajectory_dir, "iSE_contributions_before_decay.txt"), iSE_contributions)
# np.savetxt(os.path.join(trajectory_dir, "goal_contributions.txt"), goal_contributions)
# np.savetxt(os.path.join(trajectory_dir, "decay_rate.txt"), decay_rate)
# np.savetxt(os.path.join(trajectory_dir, "iSE_contributions_final.txt"), ise_contributions_final)
