import numpy as np
from Models.functions import gen_iSE_track, gen_SE_track
from Models.intentFunctions import gen_goal_driven_track, gen_gp_traj_with_goal_mean, gen_gp_bridge

#Parameter settings
d= 5
s2 = 100 #First s2 mentioned in paper
ls = 3 #Also mentioned in paper
Tmax = 100

goal = np.array([10.0,10.0])

iSE_track = gen_iSE_track(Tmax, d,s2,ls)

SE_track = gen_SE_track(Tmax, d,s2,ls)

goal_SE_track = gen_goal_driven_track(Tmax,d,s2,ls,goal)

goalMeanGP_track = gen_gp_traj_with_goal_mean(Tmax, d, s2, ls, goal)

goalConditionedGP_track = gen_gp_bridge(Tmax, s2, ls, goal)

np.savetxt("Data/iSE_track.txt", iSE_track)
np.savetxt("Data/SE_track.txt", SE_track)
np.savetxt("Data/goal_SE_track.txt", goal_SE_track)
np.savetxt("Data/goalMeanGP_track.txt", goalMeanGP_track)
np.savetxt("Data/goalConditionedGP_track.txt", goalConditionedGP_track)
