import seaborn as sns
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataFunctions import get_rmse_data_for_visualization
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm


def load_goal_predictions(filename):
    """
    Load goal predictions from a text file.
    
    Parameters:
        filename (str): Path to the text file containing goal predictions
        
    Returns:
        np.ndarray: Array of shape (T, 2) with goal coordinates at each time step
    """
    try:
        # Load the data
        goal_predictions = np.loadtxt(filename)
        
        # Ensure it's 2D
        if goal_predictions.ndim == 1:
            goal_predictions = goal_predictions.reshape(-1, 2)
        
        print(f"Loaded {len(goal_predictions)} goal predictions from {filename}")
        return goal_predictions
        
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None



name_options = ["best_goals_-20_20.txt", "best_goals_20_20.txt", "best_goals_20_-20.txt", "best_goals_-20_-20.txt"]
goal_options = ["-20 20", "20 20", "20 -20", "-20 -20"]

# Define goal options and create mapping to indices
goal_options_coords = [np.array([-20, 20]), np.array([20, 20]), np.array([20, -20]), np.array([-20, -20])]
goal_to_index = {tuple(goal): i for i, goal in enumerate(goal_options_coords)}
index_to_goal = {i: goal for i, goal in enumerate(goal_options_coords)}

print("Goal mapping:")
for i, goal in enumerate(goal_options_coords):
    print(f"Index {i}: {goal}")

# Load goal predictions and convert to indices
all_goal_vectors = {}
base_path = "Debugging/SE_MultTargets/Quad/Quad[ 20 -20]_8/SG:2.0/Gvar:10"

# Check if the base path exists
if not os.path.exists(base_path):
    print(f"Warning: Base path {base_path} does not exist!")
    print("Available paths in Debugging/SE_MultTargets/Quad/:")
    if os.path.exists("Debugging/SE_MultTargets/Quad/"):
        for item in os.listdir("Debugging/SE_MultTargets/Quad/"):
            print(f"  {item}")
    exit()

for name_option, goal_coord in zip(name_options, goal_options_coords):
    goal_index = goal_to_index[tuple(goal_coord)]
    file_path = f"{base_path}/{name_option}"
    print(f"Loading file: {file_path}")
    all_goal_vectors[goal_index] = load_goal_predictions(file_path)

# Check if all files were loaded successfully
missing_files = []
for i in range(len(goal_options_coords)):
    if all_goal_vectors[i] is None:
        missing_files.append(name_options[i])

if missing_files:
    print(f"Error: Could not load the following files:")
    for file in missing_files:
        print(f"  {file}")
    print("Please check that the files exist and the path is correct.")
    exit()

num_goal_options = 4
num_time_steps = min(len(all_goal_vectors[0]), 100)  # Use actual data length or 100, whichever is smaller

for i in range(num_goal_options):
    # Reset the plot matrix to zeros for each goal
    plot_matrix = np.zeros([num_goal_options, num_time_steps])
    
    for k in range(num_time_steps):
        if k < len(all_goal_vectors[i]):  # Check bounds
            goal_index = int(all_goal_vectors[i][k][0])
            plot_matrix[goal_index, k] = 1

    plt.figure(figsize=(17,2.5))
    plt.imshow(plot_matrix, cmap='Purples', aspect='auto', origin='lower')

    plt.xlabel("Time Step")
    plt.xticks(np.arange(0, num_time_steps, 10)) # Show ticks every 10 time steps
    plt.xlim(-0.5, num_time_steps - 0.5) # Ensure full cells are visible

    # Set Y-axis (Goal Options)
    plt.ylabel("Predicted Goal Option")
    # Set y-ticks to correspond to the 4 goal options (1-4)
    plt.yticks(np.arange(num_goal_options), [f"{goal_options[i]}" for i in range(num_goal_options)])
    plt.ylim(-0.5, num_goal_options - 0.5) # Ensure full cells are visible

    # Add a title
    #plt.title(f"Goal Initialization: {goal_options[i]}")

    # Add grid lines to clearly separate the cells
    # Vertical lines only at multiples of 5
    plt.gca().set_xticks(np.arange(0, num_time_steps, 5), minor=False)
    plt.gca().set_xticks(np.arange(-.5, num_time_steps, 5), minor=True)
    plt.grid(which='minor', axis='x', color="gray", linestyle='-', linewidth=0.5)
    
    # Horizontal lines only between goal options (not within each option)
    plt.gca().set_yticks(np.arange(-.5, num_goal_options, 1), minor=True)
    plt.grid(which='minor', axis='y', color="gray", linestyle='-', linewidth=0.5)

    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()