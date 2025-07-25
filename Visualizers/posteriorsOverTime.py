"""
Script to parse posteriors files and create a dictionary of posteriors over time
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
plt.rcParams.update({'font.size': 15})



def parse_posteriors_file(filename):
    """
    Parse the posteriors file and create a dictionary where:
    - keys are goal names in the form "[value value]" (cleaned of extra spaces)
    - values are arrays of length Tmax with posteriors at each time step
    
    Returns: dictionary with goal names as keys and posterior arrays as values
    """
    posteriors = {}
    time_steps = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by time steps
    sections = content.split("Time Step = ")
    
    for section in sections[1:]:  # Skip the first empty section
        lines = section.strip().split('\n')
        time_step = int(lines[0])
        time_steps.append(time_step)
        
        # Parse each goal's posterior for this time step
        for line in lines[1:]:
            if line.startswith("  ") and ":" in line:
                # Extract goal name and posterior value
                goal_name_unstripped = line.split(":")[0]
                goal_name = goal_name_unstripped.strip()
                posterior = float(line.split(":")[1].strip())
                
                # Clean goal name: remove extra spaces at the beginning
                goal_name = goal_name.split("[")[1]
                goal_name = goal_name.split("]")[0]
                goal_name = goal_name.strip()
                goal_x = goal_name.split(" ")[0] 
                goal_y = goal_name.split(" ")[1]
                goal_name = f"[{goal_x} {goal_y}]"
                #goal_name = f"[{goal_x} {goal_y}]"
                
                # Initialize array for this goal if not exists
                if goal_name not in posteriors:
                    posteriors[goal_name] = []
                
                # Add posterior for this time step
                posteriors[goal_name].append(posterior)
    
    # Convert lists to numpy arrays
    Tmax = len(time_steps)
    for goal_name in posteriors:
        posteriors[goal_name] = np.array(posteriors[goal_name])
    
    return posteriors, Tmax

def plot_posteriors(posteriors, Tmax):
    plt.figure(figsize=(7, 6))
    for goal_name in posteriors:
        plt.plot(np.arange(50), posteriors[goal_name][:50], label=goal_name, linewidth=2.5)
    plt.xlabel("Time Step", fontweight='bold')
    plt.ylabel("Posterior Probability", fontweight='bold')
    plt.legend()
    plt.show()

def main():
    filename = "Debugging/SE_MultTargets/CSG_PED/1_[35 90]_S2:100/SG:0.2/Gvar:10/posteriors.txt"
    posteriors, Tmax = parse_posteriors_file(filename)

    plot_posteriors(posteriors, Tmax)

if __name__ == "__main__":
    main()