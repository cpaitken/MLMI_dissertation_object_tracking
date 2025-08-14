## File to visualise individual UZH trajectory ##


import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
from dataFunctions import make_groundtruth, save_trajectory_plot
from pathlib import Path

########################################################
## Get specific UZH trajectory file and make groundtruth ##
UZH = True
cur_num = 2
groundtruth_file = f"Data/UZH/Easy/UZH_{cur_num}.txt"

groundtruth = make_groundtruth(groundtruth_file, UZH=UZH)
groundtruth = groundtruth[::20]


########################################################
## Plot and save in one UZH folder ##
plot_folder = "Data/UZH/plots"

save_trajectory_plot(groundtruth, f"UZH_{cur_num}_Easy.png", plot_folder, true_goal=None, show_start_end=True)







