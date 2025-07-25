import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import numpy as np
import re
import os

def parse_results_file(file_path):
    """Parse the final_results.txt file and extract RMSE and distance to goal data."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"Total content length: {len(content)}")
    
    # Initialize dictionaries for each G-var
    data = {}
    
    # Use regex to find all parameter combinations
    # Pattern to match: SG:X_GV:Y: followed by content until next SG: or end of file
    pattern = r'SG:([\d.]+)_GV:(\d+):(.*?)(?=SG:[\d.]+_GV:\d+:|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    print(f"Found {len(matches)} parameter combinations")
    
    for i, (sg_str, gv_str, section_content) in enumerate(matches):
        print(f"\n--- Processing combination {i+1} ---")
        
        sg = float(sg_str)
        gv = int(gv_str)
        print(f"SG: {sg}, GV: {gv}")
        print("Section content length:", len(section_content))
        
        # Extract Overall RMSE
        rmse_match = re.search(r'Overall RMSE: ([\d.]+)', section_content)
        if rmse_match:
            rmse = float(rmse_match.group(1))
            print("RMSE:", rmse)
        else:
            print("no rmse found")
            continue
            
        # Extract Change to Normal RMSE
        normal_rmse_match = re.search(r'Change to Normal RMSE: ([\d.]+)', section_content)
        if normal_rmse_match:
            normal_rmse = float(normal_rmse_match.group(1))
            print("Normal RMSE:", normal_rmse)
        else:
            print("no normal rmse found")
            normal_rmse = None
            
        # Extract Distance to Goal
        dist_match = re.search(r'Distance to Goal: ([\d.]+)', section_content)
        if dist_match:
            distance = float(dist_match.group(1))
            print("Distance:", distance)
        else:
            print("no distance found")
            continue
            
        # Store data grouped by G-var, with sigma_g as key
        if gv not in data:
            data[gv] = {}
        
        data[gv][sg] = {'rmse': rmse, 'distance': distance, 'normal_rmse': normal_rmse}
        print(f"Added data for SG:{sg}, GV:{gv}")
    
    print(f"\nFinal data structure:")
    for gv in sorted(data.keys()):
        print(f"G-var {gv}: {len(data[gv])} entries")
    
    return data

def create_plots(data, output_path=None, show_legend=False):
    """Create plots for each G-var group showing RMSE and distance to goal vs sigma_g."""
    
    # Sort G-var values for consistent plotting order and filter to include 0, 10, and 50
    gv_values = sorted([gv for gv in data.keys() if gv in [0, 10, 50]])
    
    # Create subplots for each G-var value
    fig, axes = plt.subplots(1, len(gv_values), figsize=(6*len(gv_values), 4))
    if len(gv_values) == 1:
        axes = [axes]
    
    for i, gv in enumerate(gv_values):
        ax = axes[i]
        
        # Get data for this G-var
        gv_data = data[gv]
        
        # Extract sigma_g values and corresponding metrics
        sg_values = sorted(gv_data.keys())
        rmse_values = [gv_data[sg]['rmse'] for sg in sg_values]
        distance_values = [gv_data[sg]['distance'] for sg in sg_values]
        normal_rmse_values = [gv_data[sg]['normal_rmse'] for sg in sg_values if gv_data[sg]['normal_rmse'] is not None]
        
        # Create primary y-axis for RMSE
        color1 = 'darkblue'
        ax.set_xlabel('Goal Process Variance')
        # Only show RMSE label on the leftmost plot
        if i == 0:
            ax.set_ylabel('RMSE', color=color1, fontsize=14, fontweight='bold')
        else:
            ax.set_ylabel('', color=color1)
        line1 = ax.plot(sg_values, rmse_values, 'o-', color=color1, label='Overall RMSE', linewidth=2, markersize=6)
        ax.tick_params(axis='y', labelcolor=color1)
        
        # Add Normal RMSE line if available
        if normal_rmse_values and len(normal_rmse_values) == len(sg_values):
            line3 = ax.plot(sg_values, normal_rmse_values, '--', color='tab:green', label='Normal RMSE', linewidth=2)
            lines = line1 + line3
        else:
            lines = line1
        
        # Create secondary y-axis for Distance to Goal
        ax2 = ax.twinx()
        color2 = 'tab:red'
        # Only show Distance to Goal label on the rightmost plot
        if i == len(gv_values) - 1:
            ax2.set_ylabel('Distance to Goal (m)', color=color2, fontsize=14, fontweight='bold')
        else:
            ax2.set_ylabel('', color=color2)
        line2 = ax2.plot(sg_values, distance_values, 's-', color=color2, label='Distance to Goal', linewidth=2, markersize=6)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add title
        ax.set_title(f'Goal Initialisation Variance = {gv}', fontsize=15, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend only if show_legend is True
        if show_legend:
            lines = lines + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=12)
    
    plt.tight_layout(w_pad=2.0, h_pad=0.5)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()

def main():
    # Path to the results file

    #lambda_options = ["0.01", "0.02", "0.03", "0.05"]
    file_options = ["TrueGoal_TrueLambda", "TrueGoal_FalseLambda", "FalseGoal_TrueLambda", "FalseGoal_FalseLambda"]
    lambda_options = ["0.05"]
    #file_options = ["TrueGoal_TrueLambda"]
    results_file = "Debugging/conv_iSE/basic_debugging/knownLambda_detailDebug/TrueGoal_TrueLambda/final_results.txt"
    
    # Parse the data
    for lambda_val in lambda_options:
        results_folder = f"Debugging/conv_iSE/basic_debugging/knownLambda_detailDebug"
        for file_option in file_options:
            results_file = os.path.join(results_folder, file_option, "final_results.txt")
            data = parse_results_file(results_file)
            output_path = os.path.join(results_folder, file_option, "giSE_parameter_analysis.png")
            # Only show legend for TrueGoal_TrueLambda with lambda=0.01
            show_legend = (lambda_val == "0.01" and file_option == "TrueGoal_TrueLambda")
            create_plots(data, output_path, show_legend)

    # data = parse_results_file(results_file)
    
    # # Print summary of parsed data
    # print("Parsed data summary:")
    # for gv in sorted(data.keys()):
    #     print(f"G-var {gv}: {len(data[gv])} data points")
    #     print(f"  Sigma G values: {sorted(data[gv].keys())}")
    #     rmse_vals = [data[gv][sg]['rmse'] for sg in data[gv].keys()]
    #     dist_vals = [data[gv][sg]['distance'] for sg in data[gv].keys()]
    #     print(f"  RMSE range: {min(rmse_vals):.3f} - {max(rmse_vals):.3f}")
    #     print(f"  Distance range: {min(dist_vals):.3f} - {max(dist_vals):.3f}")
    #     print()
    
    # # Create plots
    # create_plots(data, "giSE_parameter_analysis.png")

if __name__ == "__main__":
    main()
