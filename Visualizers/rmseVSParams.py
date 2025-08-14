#!/usr/bin/env python3
## File to plot the RMSE over sigma_g and G_var values for gSE model ##
## Figure 4.12 ##


import numpy as np
import matplotlib.pyplot as plt
import re
import os

def parse_convergence_statistics(filename):
    rmse_data = {}
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find all parameter sections
    sections = content.split('Parameters: sigma_g=')
    
    for section in sections[1:]:  # Skip the first empty section
        # Extract sigma_g and G_var
        sigma_g_match = re.search(r'(\d+\.?\d*), G_var=(\d+)', section)
        if sigma_g_match:
            sigma_g = float(sigma_g_match.group(1))
            g_var = int(sigma_g_match.group(2))
            
            # Extract mean and std RMSE
            rmse_match = re.search(r'Mean Posterior Weighted RMSE: ([\d.]+) ± ([\d.]+)', section)
            if rmse_match:
                mean_rmse = float(rmse_match.group(1))
                std_rmse = float(rmse_match.group(2))
                
                rmse_data[(sigma_g, g_var)] = (mean_rmse, std_rmse)
    
    return rmse_data

def plot_rmse_vs_sigma_g(rmse_data, save_path=None):
    """
    Create a line plot of RMSE vs sigma_g with different colors for each G_var
    """
    # Extract unique sigma_g and G_var values
    sigma_g_values = sorted(list(set(key[0] for key in rmse_data.keys())))
    g_var_values = sorted(list(set(key[1] for key in rmse_data.keys())))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Color palette for different G_var values
    colors = plt.cm.tab10(np.linspace(0, 1, len(g_var_values)))
    
    for i, g_var in enumerate(g_var_values):
        # Extract data for this G_var
        x_values = []
        y_values = []
        y_errors = []
        
        for sigma_g in sigma_g_values:
            if (sigma_g, g_var) in rmse_data:
                mean_rmse, std_rmse = rmse_data[(sigma_g, g_var)]
                x_values.append(sigma_g)
                y_values.append(mean_rmse)
                y_errors.append(std_rmse)
        
        # Plot line with error bars
        plt.errorbar(x_values, y_values, yerr=y_errors, 
                    marker='o', linewidth=2, markersize=6,
                    color=colors[i], label=f'G_var = {g_var}',
                    capsize=5, capthick=1)
    
    # Customize the plot
    plt.xlabel('σ_g (Goal Process Noise)', fontsize=15)
    plt.ylabel('Mean RMSE', fontsize=15)
    #plt.title('Effect of Process Noise and Goal Variance on Tracking RMSE', fontsize=17, fontweight='bold')
    plt.legend(title='G_var Values', fontsize=15)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to log scale if sigma_g values span multiple orders of magnitude
    # if max(sigma_g_values) / min(sigma_g_values) > 10:
    #     plt.xscale('log')
    #     plt.xlabel('σ_g (Goal Process Noise) - Log Scale', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def create_summary_table(rmse_data):
    print("RMSE Summary by Parameter Combination:")
    print("=" * 60)
    print(f"{'σ_g':<8} {'G_var':<8} {'Mean RMSE':<12} {'Std RMSE':<12}")
    print("-" * 60)
    
    # Sort by sigma_g, then by G_var
    sorted_keys = sorted(rmse_data.keys(), key=lambda x: (x[0], x[1]))
    
    for sigma_g, g_var in sorted_keys:
        mean_rmse, std_rmse = rmse_data[(sigma_g, g_var)]
        print(f"{sigma_g:<8.1f} {g_var:<8} {mean_rmse:<12.6f} {std_rmse:<12.6f}")

def main():
    # File path - adjust this to match your file location
    filename = "Debugging/SE_MultTargets/Quad_PED/Convergence_statistics.txt"
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        print("Please update the filename variable in the script.")
        return
    
    # Parse the data
    rmse_data = parse_convergence_statistics(filename)
    
    if not rmse_data:
        print("No RMSE data found in the file!")
        return
    
    # Print summary table
    create_summary_table(rmse_data)
    
    # Create and save the plot
    save_path = "Debugging/SE_MultTargets/Quad_PED/rmse_analysis_plot.png"
    plot_rmse_vs_sigma_g(rmse_data, save_path)
    
    # Print some insights
    print("\n" + "=" * 60)
    print("INSIGHTS:")
    print("=" * 60)
    
    # Find best and worst performing combinations
    best_combo = min(rmse_data.items(), key=lambda x: x[1][0])
    worst_combo = max(rmse_data.items(), key=lambda x: x[1][0])
    
    print(f"Best performing combination: σ_g={best_combo[0][0]}, G_var={best_combo[0][1]}")
    print(f"  Mean RMSE: {best_combo[1][0]:.6f} ± {best_combo[1][1]:.6f}")
    
    print(f"Worst performing combination: σ_g={worst_combo[0][0]}, G_var={worst_combo[0][1]}")
    print(f"  Mean RMSE: {worst_combo[1][0]:.6f} ± {worst_combo[1][1]:.6f}")
    
    # Calculate overall statistics
    all_rmse_means = [data[0] for data in rmse_data.values()]
    print(f"\nOverall mean RMSE: {np.mean(all_rmse_means):.6f} ± {np.std(all_rmse_means):.6f}")

if __name__ == "__main__":
    main()