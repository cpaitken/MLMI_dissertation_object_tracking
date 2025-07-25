import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

def parse_pf_results(base_path="Debugging/conv_iSE/varyingLambda/fixedLamTrajectories"):
    """
    Parse particle filter results from the folder hierarchy and organize into a dictionary.
    
    Returns:
        dict: Nested dictionary with structure:
        {
            'FalseGoal_FalseLambda': {
                0.01: {'mean_rmse': float, 'std_rmse': float, 'mean_distance': float, 
                       'std_distance': float, 'mean_time': float, 'std_time': float,
                       'mean_final_lambda': float, 'std_final_lambda': float},
                0.02: {...},
                ...
            },
            'FalseGoal_TrueLambda': {...},
            ...
        }
    """
    results_dict = {}
    
    # Define the run types and lambda values
    run_types = ['FalseGoal_FalseLambda', 'FalseGoal_TrueLambda', 
                 'TrueGoal_FalseLambda', 'TrueGoal_TrueLambda']
    lambda_values = [0.01, 0.02, 0.03, 0.05]
    
    for run_type in run_types:
        results_dict[run_type] = {}
        
        for lambda_val in lambda_values:
            # Construct the path to the overall_results.txt file
            lambda_folder = f"lambda_{lambda_val}"
            results_file_path = os.path.join(base_path, run_type, lambda_folder, "overall_results.txt")
            
            # Initialize metrics dictionary
            metrics = {
                'mean_rmse': 0.0, 'std_rmse': 0.0,
                'mean_distance': 0.0, 'std_distance': 0.0,
                'mean_time': 0.0, 'std_time': 0.0,
                'mean_final_lambda': 0.0, 'std_final_lambda': 0.0
            }
            
            # Read the results file if it exists
            if os.path.exists(results_file_path):
                print(f"Reading: {results_file_path}")
                with open(results_file_path, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if line.startswith('Mean RMSE:'):
                        metrics['mean_rmse'] = float(line.split(':')[1].strip())
                    elif line.startswith('Std RMSE:'):
                        metrics['std_rmse'] = float(line.split(':')[1].strip())
                    elif line.startswith('Mean Distance to Goal:'):
                        metrics['mean_distance'] = float(line.split(':')[1].strip())
                    elif line.startswith('Std Distance to Goal:'):
                        metrics['std_distance'] = float(line.split(':')[1].strip())
                    elif line.startswith('Mean Time to Goal:'):
                        metrics['mean_time'] = float(line.split(':')[1].strip())
                    elif line.startswith('Std Time to Goal:'):
                        metrics['std_time'] = float(line.split(':')[1].strip())
                    elif line.startswith('Mean Final Lambda:'):
                        metrics['mean_final_lambda'] = float(line.split(':')[1].strip())
                    elif line.startswith('Std Final Lambda:'):
                        metrics['std_final_lambda'] = float(line.split(':')[1].strip())
            else:
                print(f"File not found: {results_file_path}")
            
            results_dict[run_type][lambda_val] = metrics
    
    return results_dict

def plot_pf_results(results_dict):
    """
    Create 4 separate plots (one for each lambda value), each with 4 subplots showing bar charts
    for different metrics across run types.
    """
    run_types = list(results_dict.keys())
    lambda_values = sorted(list(results_dict[run_types[0]].keys()))
    
    # Define colors and labels
    colors = ['firebrick', 'darkorange', 'royalblue', 'seagreen']
    run_labels = ['FF', 'FT', 'TF', 'TT']
    
    # Create 4 separate figures, one for each lambda value
    for lambda_val in lambda_values:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'Lambda = {lambda_val}', fontsize=16, fontweight='bold')
        
        # Get data for this lambda value
        rmse_values = [results_dict[run_type][lambda_val]['mean_rmse'] for run_type in run_types]
        rmse_stds = [results_dict[run_type][lambda_val]['std_rmse'] for run_type in run_types]
        distance_values = [results_dict[run_type][lambda_val]['mean_distance'] for run_type in run_types]
        distance_stds = [results_dict[run_type][lambda_val]['std_distance'] for run_type in run_types]
        time_values = [results_dict[run_type][lambda_val]['mean_time'] for run_type in run_types]
        time_stds = [results_dict[run_type][lambda_val]['std_time'] for run_type in run_types]
        final_lambda_values = [results_dict[run_type][lambda_val]['mean_final_lambda'] for run_type in run_types]
        final_lambda_stds = [results_dict[run_type][lambda_val]['std_final_lambda'] for run_type in run_types]
        
        # Plot 1: Mean RMSE
        ax1 = axes[0, 0]
        bars1 = ax1.bar(run_labels, rmse_values, yerr=rmse_stds, color=colors, alpha=0.7, 
                       edgecolor='black', capsize=3, width=0.6)
        ax1.set_ylabel('Mean RMSE')
        ax1.set_title('Mean RMSE')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean Distance to Goal
        ax2 = axes[0, 1]
        bars2 = ax2.bar(run_labels, distance_values, yerr=distance_stds, color=colors, alpha=0.7, 
                       edgecolor='black', capsize=3, width=0.6)
        ax2.set_ylabel('Mean Distance to Goal')
        ax2.set_title('Mean Distance to Goal')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mean Time to Goal
        ax3 = axes[1, 0]
        bars3 = ax3.bar(run_labels, time_values, yerr=time_stds, color=colors, alpha=0.7, 
                       edgecolor='black', capsize=3, width=0.6)
        ax3.set_ylabel('Mean Time to Goal')
        ax3.set_title('Mean Time to Goal')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Mean Final Lambda
        ax4 = axes[1, 1]
        bars4 = ax4.bar(run_labels, final_lambda_values, yerr=final_lambda_stds, color=colors, alpha=0.7, 
                       edgecolor='black', capsize=3, width=0.6)
        ax4.set_ylabel('Mean Final Lambda')
        ax4.set_title('Mean Final Lambda')
        ax4.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save each plot separately
        plt.savefig(f'PF_results_lambda_{lambda_val}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return fig

# Example usage:
if __name__ == "__main__":
    # Parse the results
    results = parse_pf_results()
    
    # Create the plots
    fig = plot_pf_results(results)
    
    # Save the plot
    #plt.savefig('PF_results_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
