#!/usr/bin/env python3
## File to plot goal options for quadrant dataset ##
## Figure 4.2 ##
## Base structure written by ChatGPT, then personally edited ##

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
import matplotlib.patches as patches

def create_quadrant_goal_plot():
    """
    Create a plot showing the 4 goal options for the quadrant dataset.
    Each goal has 25% prior probability and is represented by a red box.
    """
    # Define the 4 goal locations (corners of a square)
    goal_locations = np.array([
        [20, 20],    # Top right
        [-20, 20],   # Top left  
        [-20, -20],  # Bottom left
        [20, -20]    # Bottom right
    ])
    
    # All goals have 25% prior probability
    prior_probabilities = [0.25, 0.25, 0.25, 0.25]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot each goal as a colored box
    for i, (goal, prob) in enumerate(zip(goal_locations, prior_probabilities)):
        # Create a rectangle for each goal
        # Box size: 8x8 centered on the goal location
        x, y = goal
        rect = patches.Rectangle((x-2.5, y-2.5), 5, 5, 
                                linewidth=1, 
                                edgecolor='black', 
                                facecolor='red', 
                                alpha=0.6)
        ax.add_patch(rect)
        
        # Add coordinate labels above each box
        ax.text(x, y + 6, f'{x}, {y}', 
                ha='center', va='bottom', fontsize=17, fontweight='bold')
        
        # Add probability label below each box
        #ax.text(x, y - 6, f'{prob:.0%}', 
        #        ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Set plot properties
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_xlabel('X', fontsize=15, fontweight='bold')
    ax.set_ylabel('Y', fontsize=15, fontweight='bold')
    #ax.set_title('Quadrant Goal Options: Locations and Prior Probabilities', 
    #             fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [patches.Patch(color='red', alpha=0.6, label='25% Prior Probability')]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=17)
    
    # Make axes equal aspect ratio
    ax.set_aspect('equal')
    
    # Add origin marker
    ax.plot(0, 0, 'ko', markersize=8, label='Origin')
    ax.text(2, 2, 'Trajectory Origin', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('quadrant_goal_options.png', dpi=300, bbox_inches='tight')
    print("Saved plot: quadrant_goal_options.png")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    create_quadrant_goal_plot() 