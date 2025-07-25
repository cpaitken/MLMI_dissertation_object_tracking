"""
Script to analyze convergence statistics and plot RMSE vs sigma_g for different G_var values
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import re
import os

def parse_convergence_statistics(filename):
    """
    Parse the convergence statistics text file and extract RMSE data
    Returns: dictionary with (sigma_g, G_var) as keys and (mean_rmse, std_rmse) as values
    """
    base_rmse_data = {}
    sep_tracking_rmse_data = {}
    
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
            base_rmse_match = re.search(r'Mean Posterior Weighted RMSE: ([\d.]+) ± ([\d.]+)', section)
            if base_rmse_match:
                mean_rmse = float(base_rmse_match.group(1))
                std_rmse = float(base_rmse_match.group(2))
                
                base_rmse_data[(sigma_g, g_var)] = (mean_rmse, std_rmse)
            
            sep_tracking_rmse_match = re.search(r'Best Overall Prediction RMSE: ([\d.]+) ± ([\d.]+)', section)
            if sep_tracking_rmse_match:
                mean_rmse = float(sep_tracking_rmse_match.group(1))
                std_rmse = float(sep_tracking_rmse_match.group(2))
                
                sep_tracking_rmse_data[(sigma_g, g_var)] = (mean_rmse, std_rmse)
    
    return base_rmse_data, sep_tracking_rmse_data


def plot_comparison_bar_charts(base_rmse_data, sep_tracking_rmse_data):
    """
    Create comparison bar charts: two subplots separated by G_var, 
    each showing base RMSE vs tracking RMSE for different sigma_g values
    """
    # Get unique values
    sigma_g_values = sorted(list(set([key[0] for key in base_rmse_data.keys()])))
    g_var_values = sorted(list(set([key[1] for key in base_rmse_data.keys()])))
    g_var_values = g_var_values[:2]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(g_var_values), figsize=(5*len(g_var_values), 6))
    if len(g_var_values) == 1:
        axes = [axes]
    
    # Colors for the bars
    base_color = 'skyblue'
    tracking_color = 'lightcoral'


    
    for i, g_var in enumerate(g_var_values):
        ax = axes[i]
        
        # Prepare data for this G_var
        base_means = []
        base_stds = []
        tracking_means = []
        tracking_stds = []
        
        for sigma_g in sigma_g_values:
            if (sigma_g, g_var) in base_rmse_data:
                base_mean, base_std = base_rmse_data[(sigma_g, g_var)]
                base_means.append(base_mean)
                base_stds.append(base_std)
            else:
                base_means.append(0)
                base_stds.append(0)
                
            if (sigma_g, g_var) in sep_tracking_rmse_data:
                tracking_mean, tracking_std = sep_tracking_rmse_data[(sigma_g, g_var)]
                tracking_means.append(tracking_mean)
                tracking_stds.append(tracking_std)
            else:
                tracking_means.append(0)
                tracking_stds.append(0)
        
        # Set up bar positions
        x = np.arange(len(sigma_g_values))
        width = 0.35
        
        # Create bars
        base_bars = ax.bar(x - width/2, base_means, width, label='Single State Set', 
                          color=base_color, alpha=0.8, yerr=base_stds, capsize=5)
        tracking_bars = ax.bar(x + width/2, tracking_means, width, label='Separate State Sets', 
                              color=tracking_color, alpha=0.8, yerr=tracking_stds, capsize=5)
        
        # Customize the plot
        ax.set_xlabel('Goal Process Variance', fontsize=15, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=15, fontweight='bold')
        ax.set_title(f'G_var = {g_var}', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{sg:.1f}' for sg in sigma_g_values], fontsize=13)
        ax.legend(loc=4)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in base_bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 0.75,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        for bar in tracking_bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 0.75,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def main():
    filename = "Debugging/SE_MultTargets/CSG_PED/Convergence_statistics.txt"
    base_rmse_data, sep_tracking_rmse_data = parse_convergence_statistics(filename)
    plot_comparison_bar_charts(base_rmse_data, sep_tracking_rmse_data)

if __name__ == "__main__":
    main()
