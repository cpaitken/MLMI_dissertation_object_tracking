import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from Models.functions import gen_iSE_track, gen_SE_track
from Models.intentFunctions import gen_goal_driven_track, gen_gp_traj_with_goal_mean, gen_gp_bridge, gen_iSE_driven_track

#Parameter settings
d= 5
s2 = 100 #First s2 mentioned in paper
ls = 3 #Also mentioned in paper
Tmax = 100

goal = np.array([10.0,10.0])

iSE_track = gen_iSE_track(Tmax, d,s2,ls)

SE_track = gen_SE_track(Tmax, d,s2,ls)

goal_SE_track = gen_goal_driven_track(Tmax,d,s2,ls,goal)
header_driven_track = f"d={d}, s2={s2}, l={ls}, goal={goal.tolist()}"

goalMeanGP_track = gen_gp_traj_with_goal_mean(Tmax, d, s2, ls, goal)

goalConditionedGP_track = gen_gp_bridge(Tmax, s2, ls, goal)

#Trying with a different s2 and l3
goalMeanGP_track_s1 = gen_gp_traj_with_goal_mean(Tmax, d, 1, ls, goal)

iSE_goal_driven_track = gen_iSE_driven_track(Tmax, d, s2, ls, goal)
header_ise_track = f"d={d}, s2={s2}, l={ls}, goal={goal.tolist()}"

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data/Generated'))
os.makedirs(data_dir, exist_ok=True)

np.savetxt(os.path.join(data_dir, "iSE_track.txt"), iSE_track)
np.savetxt(os.path.join(data_dir, "SE_track.txt"), SE_track)
np.savetxt(os.path.join(data_dir, "goal_SE_track.txt"), goal_SE_track, header=header_driven_track)
np.savetxt(os.path.join(data_dir, "goalMeanGP_track.txt"), goalMeanGP_track)
np.savetxt(os.path.join(data_dir, "goalConditionedGP_track.txt"), goalConditionedGP_track)
np.savetxt(os.path.join(data_dir, "goalMeanGP_track_s1.txt"), goalMeanGP_track_s1)
np.savetxt(os.path.join(data_dir, "goal_iSE_track.txt"), iSE_goal_driven_track, header=header_ise_track)
