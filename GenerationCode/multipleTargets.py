## File used to generate either CSG or Quadrant datasets ##
## Specify if CSG or Quadrant in line 16 ##
## Currently generating for CSG dataset ##

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from scipy.spatial.distance import cdist
import matplotlib.patches as patches
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.intentFunctions import gen_goal_driven_track, gen_gp_bridge

#Set to True for CSG, False for Quad
csg = True

quad_goal_options = [np.array([-20,20]), np.array([20,20]), np.array([20,-20]), np.array([-20,-20])]
quad_goal_indices = [0,1,2,3]
quad_goal_probs = [0.25, 0.25, 0.25, 0.25]

csg_goal_options = [np.array([50,100]), np.array([35,115]), np.array([65,115]), np.array([35, 90]), 
np.array([65, 90]), np.array([50, 85]), np.array([35, 65]), np.array([65, 65]),
np.array([50, 60]), np.array([35, 45]), np.array([65, 45])]
csg_goal_indices = [0,1,2,3,4,5,6,7,8,9,10]
csg_goal_probs = [0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05]

if csg:
    goal_options = csg_goal_options
    goal_indices = csg_goal_indices
    goal_probs = csg_goal_probs
else:
    goal_options = quad_goal_options
    goal_indices = quad_goal_indices
    goal_probs = quad_goal_probs

#Create a function to generate a goal based on the goal_options and goal_probs

#Parameters
Tmax = 150
d = 5
s2 = 100
ls = 5
usingGP_notBridge = False

def generate_goal():
    return np.random.choice(goal_indices, p=goal_probs)

# Plot goal options with color-coded rectangles
plt.figure(figsize=(12, 8))

# Define colors based on probability
colors = []
for i, prob in enumerate(csg_goal_probs):
    if i == 0:  # goal_option[0] - bright red
        colors.append('red')
    elif 1 <= i < 6:  # goal_option[1-6] - orange
        colors.append('orange')
    else:  # goal_option[7-11] - yellow
        colors.append('yellow')

# Plot rectangles for each goal option
for i, goal in enumerate(goal_options):
    x, y = goal[0], goal[1]
    
    # Set dimensions based on goal position
    if np.allclose(goal, [50, 100]):
        width, height = 3, 8
    else:
        width, height = 2, 5
    
    # Create rectangle
    rect = plt.Rectangle((x - width/2, y - height/2), width, height, 
                        facecolor=colors[i], alpha=0.7, edgecolor='black', linewidth=1)
    plt.gca().add_patch(rect)
    
    # Add text label
    plt.text(x, y + height/2 + 0.25, f'{x}, {y}', ha='center', va='bottom', 
             fontsize=14, fontweight='bold')

plt.xlabel('X', fontweight='bold')
plt.ylabel('Y', fontweight='bold')
#plt.title('Complex Goal Options: Locations and Prior Probabilities')
plt.grid(True, alpha=0.3)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.axis('equal')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.7, label='25%'),
    Patch(facecolor='orange', alpha=0.7, label='10%'),
    Patch(facecolor='yellow', alpha=0.7, label='5%')
]
plt.legend(handles=legend_elements, title='Prior Probability', loc='upper right', fontsize=17)

plt.show()

# for i in range(1):
#     #goal = goal_options[generate_goal()]
#     goal = goal_options[0]
#     print(goal)

#     # Generate random start position along x-axis between 0 and 100
#     if csg:
#         start_x = np.random.uniform(0, 100)
#         start = np.array([start_x, 0])
#     else:
#         start = np.array([0,0])
#     if csg:
#         run_name = str(i) + "_" + str(goal) + "_S2:" + str(s2) + "_LS:" + str(ls)
#     else:
#         run_name = str(i) + "_" + str(goal) + "_S2:" + str(s2) + "_LS:" + str(ls)
#     output_folder = f"Data/Generated/Parameter_Study/{run_name}"
#     os.makedirs(output_folder, exist_ok=True)

#     if usingGP_notBridge:
#         track = gen_goal_driven_track(Tmax, d, s2, ls, goal)
#     else:
#         track = gen_gp_bridge(Tmax, s2, ls, goal, dt=1, start=start)
#     np.savetxt(os.path.join(output_folder, "track.txt"), track, fmt="%.6f", header="x y")

#     plt.plot(track[:,0], track[:,1])

#     for idx, g in enumerate(goal_options):
#         color = 'green' if np.allclose(g, goal) else 'red'
#         rect = patches.Rectangle(
#             (g[0] - 2.5, g[1] - 2.5), 5, 5,
#             linewidth=0.05,
#             edgecolor='black',
#             facecolor=color,
#             alpha=0.3,
#             zorder=2
#         )
#         plt.gca().add_patch(rect)

#     plt.scatter(*zip(*goal_options), color='black', marker='x', label='Goal options')
#     plt.scatter(goal[0], goal[1], color='green', marker='o', label='Chosen goal')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.savefig(os.path.join(output_folder, "track_plot.png"), bbox_inches='tight')
#     plt.close()

