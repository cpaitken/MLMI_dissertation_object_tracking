import os
import numpy as np
import matplotlib.pyplot as plt
import re
import ast
from scipy.interpolate import make_interp_spline
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib as mpl

__all__ = [
    "make_groundtruth",
    "pretty_print_matrix",
    "save_vector_arrays_txt",
    "save_matrix_arrays_txt",
    "save_state_comparison_txt",
    "get_model_rmse",
    "save_tracking_plot",
    "save_vector_array_txt",
    "save_matrix_array_txt",
    "save_state_array_txt",
    "save_specifications_txt",
    "save_variance_array_txt",
    "print_rmse_summary",
    "extract_rmse_by_parameters",
    "calculate_rmse_statistics_by_parameters",
    "print_rmse_statistics_by_parameters",
    "save_rmse_to_txt",
    "find_convergence_time",
    "print_convergence_summary"
]

def make_groundtruth(filename, UZH=False):
    data = np.loadtxt(filename, comments="#")

    if UZH:
        tx_ty = data[:, [1,2]]
    else:
        tx_ty = data[:, [0,1]]

    #tx_ty_list = [np.array([tx, ty]) for tx, ty in tx_ty]

    return tx_ty

def extract_params_from_header(filename):
    with open(filename, 'r') as f:
        first_line = f.readline()
        if first_line.startswith('#'):
            header = first_line[1:].strip()  # Remove '#' and whitespace
            # This regex matches key=value pairs, where value can be a list or a number
            pattern = r'(\w+)=((?:\[[^\]]*\])|(?:[^,]+))'
            matches = re.findall(pattern, header)
            params = {}
            for key, value in matches:
                key = key.strip()
                value = value.strip()
                try:
                    # Try to parse as Python literal (list, int, float, etc.)
                    value = ast.literal_eval(value)
                except Exception:
                    # Fallback: try to parse as float or int
                    value = float(value) if '.' in value else int(value)
                params[key] = value
            return params
        else:
            params = {}
            params["filename"] = filename
            return params

def pretty_print_matrix(matrix, name, file):
    file.write(f"{name} (shape {matrix.shape}):\n")
    for row in matrix:
        file.write("  " + "  ".join(f"{val:8.4f}" for val in row) + "\n")
    file.write("\n")

def save_vector_arrays_txt(arr1, arr2, filename, label1, label2, folder="Debugging"):
    """
    Save two arrays of vectors (shape: (N, d)) to a txt file in Debugging/ with readable formatting.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w') as f:
        f.write(f"{label1} and {label2} Comparison\n")
        f.write("=" * 50 + "\n\n")
        for k in range(min(len(arr1), len(arr2))):
            f.write(f"Step {k}:\n")
            f.write(f"{label1} (shape: {arr1[k].shape}):\n")
            for i, row in enumerate(arr1[k]):
                f.write(f"  {i}: {row}\n")
            f.write(f"{label2} (shape: {arr2[k].shape}):\n")
            for i, row in enumerate(arr2[k]):
                f.write(f"  {i}: {row}\n")
            f.write("\n" + "="*50 + "\n\n")

def save_matrix_arrays_txt(arr1, arr2, filename, label1, label2, folder="Debugging"):
    """
    Save two arrays of matrices (shape: (N, d, d)) to a txt file in Debugging/ with readable formatting, using pretty_print_matrix for each matrix.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w') as f:
        for k in range(min(len(arr1), len(arr2))):
            f.write(f"Step {k}:\n")
            pretty_print_matrix(arr1[k], f"{label1} Example {k+1}", f)
            pretty_print_matrix(arr2[k], f"{label2} Example {k+1}", f)
            f.write("\n" + "="*50 + "\n\n")

def save_state_comparison_txt(full_state_iSE, full_state_goal, filename, folder="Debugging"):
    """
    Save a detailed state comparison between iSE and goal model state arrays to Debugging/filename.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w') as f:
        f.write("DETAILED STATE VECTOR COMPARISON\n")
        f.write("=" * 50 + "\n\n")
        for k in range(min(len(full_state_iSE), len(full_state_goal))):
            f.write(f"Time Step {k}:\n")
            f.write("-" * 20 + "\n")
            # iSE Model State
            f.write("iSE Model State Vector (shape: {}):\n".format(full_state_iSE[k].shape))
            for i in range(full_state_iSE[k].shape[0]):
                f.write(f"  Position {i}: [{full_state_iSE[k][i,0]:.3f}, {full_state_iSE[k][i,1]:.3f}]\n")
            # Goal Model State
            f.write("Goal Model State Vector (shape: {}):\n".format(full_state_goal[k].shape))
            for i in range(full_state_goal[k].shape[0]):
                if i == full_state_goal[k].shape[0] - 1:
                    f.write(f"  GOAL {i}: [{full_state_goal[k][i,0]:.3f}, {full_state_goal[k][i,1]:.3f}]\n")
                else:
                    f.write(f"  Position {i}: [{full_state_goal[k][i,0]:.3f}, {full_state_goal[k][i,1]:.3f}]\n")
            # Compare corresponding positions
            f.write("Position Comparisons:\n")
            for i in range(min(full_state_iSE[k].shape[0], full_state_goal[k].shape[0] - 1)):
                diff = full_state_iSE[k][i] - full_state_goal[k][i]
                f.write(f"  Pos {i} diff: [{diff[0]:.3f}, {diff[1]:.3f}]\n")
            f.write("\n" + "="*50 + "\n\n")

def get_model_rmse(model_predictions, groundtruth):
    rmse_model = np.sqrt(np.mean(np.sum((model_predictions - groundtruth)**2, axis=1)))
    return rmse_model

def smooth_line(x, y, num_points=300):
    # x and y are 1D arrays of the original points
    t = np.arange(len(x))
    t_smooth = np.linspace(t.min(), t.max(), num_points)
    spline_x = make_interp_spline(t, x, k=3)(t_smooth)
    spline_y = make_interp_spline(t, y, k=3)(t_smooth)
    return spline_x, spline_y

def color_line(x, y, cmap='cubehelix', linewidth=2, alpha=1.0, label=None):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(x)-1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    lc.set_array(np.arange(len(x)))
    plt.gca().add_collection(lc)
    if label:
        plt.plot([], [], color=plt.get_cmap(cmap)(0.5), label=label)

def save_tracking_plot(groundtruth, noisy_data, X, G, XN, modelName1, modelName2, filename, folder="Debugging", show_Target=False, false_goals=None):
    """
    Plot and save the tracking results to Debugging/Plots/filename (PNG).
    """
    os.makedirs(os.path.join(folder), exist_ok=True)
    plt.figure()
    gt_x_smooth, gt_y_smooth = smooth_line(groundtruth[:,0], groundtruth[:,1])
    plt.plot(gt_x_smooth, gt_y_smooth, label='Truth')
    plt.scatter(*zip(*noisy_data), alpha=0.3, label='Noisy obs')
    X_x_smooth, X_y_smooth = smooth_line(X[:,0], X[:,1])
    plt.plot(X_x_smooth, X_y_smooth, label=modelName1, color='green')
    G_x_smooth, G_y_smooth = smooth_line(G[:,0], G[:,1])
    color_line(G_x_smooth, G_y_smooth, cmap='cividis', linewidth=2, alpha=1.0, label='Inferred goal')
    XN_x_smooth, XN_y_smooth = smooth_line(XN[:,0], XN[:,1])
    plt.plot(XN_x_smooth, XN_y_smooth, '--', label=modelName2, color='limegreen')
    
    # Draw a semi-transparent 5x5 square around the last groundtruth location
    if show_Target:
        end_x, end_y = groundtruth[-1, 0], groundtruth[-1, 1]
        square = patches.Rectangle((end_x - 2.5, end_y - 2.5), 5, 5, linewidth=0, edgecolor=None, facecolor='green', alpha=0.2, zorder=2)
        plt.gca().add_patch(square)
    
    # Draw red squares around false goal options
    if false_goals is not None:
        for false_goal in false_goals:
            false_x, false_y = false_goal[0], false_goal[1]
            false_square = patches.Rectangle((false_x - 2.5, false_y - 2.5), 5, 5, linewidth=0, edgecolor=None, facecolor='red', alpha=0.2, zorder=2)
            plt.gca().add_patch(false_square)
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.savefig(os.path.join(folder, filename), bbox_inches='tight')
    plt.close()

def save_vector_array_txt(arr, filename, label, folder="Debugging"):
    """
    Save a single array of vectors (shape: (N, d)) to a txt file in Debugging/ with readable formatting.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w') as f:
        f.write(f"{label} Vectors\n")
        f.write("=" * 50 + "\n\n")
        for k in range(len(arr)):
            f.write(f"Step {k} ({label} shape: {arr[k].shape}):\n")
            for i, row in enumerate(arr[k]):
                f.write(f"  {i}: {row}\n")
            f.write("\n")


def save_matrix_array_txt(arr, filename, label, folder="Debugging"):
    """
    Save a single array of matrices (shape: (N, d, d)) to a txt file in Debugging/ with readable formatting.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w') as f:
        for k in range(len(arr)):
            f.write(f"Step {k} ({label} shape: {arr[k].shape}):\n")
            pretty_print_matrix(arr[k], label, f)
            f.write("\n" + "="*50 + "\n\n")


def save_state_array_txt(state_array, filename, label, folder="Debugging"):
    """
    Save a detailed state array (shape: (N, d, 2)) to Debugging/filename for a single model.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w') as f:
        f.write(f"DETAILED STATE VECTOR ({label})\n")
        f.write("=" * 50 + "\n\n")
        for k in range(len(state_array)):
            f.write(f"Time Step {k}:\n")
            f.write("-" * 20 + "\n")
            f.write(f"{label} State Vector (shape: {state_array[k].shape}):\n")
            for i in range(state_array[k].shape[0]):
                f.write(f"  Position {i}: [{state_array[k][i,0]:.3f}, {state_array[k][i,1]:.3f}]\n")
            f.write("\n" + "="*50 + "\n\n")

def save_specifications_txt(folder, params, extra_info=None):
    """
    Save a specifications.txt file in the specified folder with parameters from a dictionary.
    Parameters:
        folder (str): The folder to save the file in.
        params (dict): Dictionary of parameters to save (key-value pairs).
        extra_info (dict, optional): Any extra info to include (key-value pairs).
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "specifications.txt")
    with open(path, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        if extra_info is not None:
            for key, value in extra_info.items():
                f.write(f"{key}: {value}\n")

def save_variance_array_txt(variances, filename, folder="Debugging"):
    """
    Save an array of variances (1D or 2D) to a txt file in the specified folder with readable formatting.
    Parameters:
        variances (np.ndarray or list): Array of variances, shape (N,) or (N, d)
        filename (str): Name of the file to save.
        folder (str): Folder to save the file in (default: 'Debugging').
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    variances = np.array(variances)
    with open(path, 'w') as f:
        f.write("Variance Array\n")
        f.write("=" * 50 + "\n\n")
        for k in range(len(variances)):
            if variances.ndim == 1:
                f.write(f"Step {k}: {variances[k]}\n")
            else:
                f.write(f"Step {k}: {variances[k]}\n")

def print_rmse_summary(baseline_folder):
    """
    Scans all subfolders (Easy, Medium, Hard) in the given baseline_folder, reads each specifications.txt,
    and prints the average and std dev of SE_RMSE and GSE_RMSE for each category, as well as the average and std dev
    of the percentage improvement of GSE_RMSE over SE_RMSE for each category.
    Also prints the overall statistics across all categories.
    """
    overall_se_rmses = []
    overall_gse_rmses = []
    overall_improvements = []

    categories = [d for d in os.listdir(baseline_folder) if os.path.isdir(os.path.join(baseline_folder, d))]
    for category in categories:
        se_rmses = []
        gse_rmses = []
        improvements = []
        category_path = os.path.join(baseline_folder, category)
        for dataset_folder in os.listdir(category_path):
            spec_path = os.path.join(category_path, dataset_folder, "specifications.txt")
            if not os.path.isfile(spec_path):
                continue
            with open(spec_path, 'r') as f:
                lines = f.readlines()
                perf_line = [line for line in lines if line.startswith('performance_params:')]
                if not perf_line:
                    continue
                # Parse the dictionary
                perf_dict = ast.literal_eval(perf_line[0].split('performance_params:')[1].strip())
                se_rmse = perf_dict.get('SE_RMSE', None)
                gse_rmse = perf_dict.get('GSE_RMSE', None)
                if se_rmse is not None and gse_rmse is not None:
                    se_rmses.append(se_rmse)
                    gse_rmses.append(gse_rmse)
                    overall_se_rmses.append(se_rmse)
                    overall_gse_rmses.append(gse_rmse)
                    if se_rmse != 0:
                        improvement = 100 * (se_rmse - gse_rmse) / se_rmse
                        improvements.append(improvement)
                        overall_improvements.append(improvement)
        if se_rmses:
            print(f"Category: {category}")
            print(f"  SE_RMSE: mean={np.mean(se_rmses):.4f}, std={np.std(se_rmses):.4f}")
            print(f"  GSE_RMSE: mean={np.mean(gse_rmses):.4f}, std={np.std(gse_rmses):.4f}")
            print(f"  GSE % improvement over SE: mean={np.mean(improvements):.2f}%, std={np.std(improvements):.2f}%\n")
        else:
            print(f"Category: {category} (no valid results found)")

    # Print overall statistics
    if overall_se_rmses:
        print("Overall statistics across all categories:")
        print(f"  SE_RMSE: mean={np.mean(overall_se_rmses):.4f}, std={np.std(overall_se_rmses):.4f}")
        print(f"  GSE_RMSE: mean={np.mean(overall_gse_rmses):.4f}, std={np.std(overall_gse_rmses):.4f}")
        print(f"  GSE % improvement over SE: mean={np.mean(overall_improvements):.2f}%, std={np.std(overall_improvements):.2f}%\n")
    else:
        print("No valid results found in any category.")

def save_particles_txt(particles, filename, folder="Debugging"):
    """
    Save all particles at each timestep to a txt file in a readable format.
    
    Parameters:
        particles: np.ndarray of shape (num_timesteps, num_particles, state_dim) or (num_timesteps, num_particles, d, 2)
        filename: str, name of the file to save
        folder: str, folder to save the file in (default: 'Debugging')
    """
    import os
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w') as f:
        for t, timestep_particles in enumerate(particles):
            f.write(f"Time Step {t}:\n")
            f.write("-" * 20 + "\n")
            for i, particle in enumerate(timestep_particles):
                f.write(f"  Particle {i}: {np.array2string(particle, precision=4, separator=', ')}\n")
            f.write("\n" + "="*50 + "\n\n")

def extract_rmse_by_parameters(base_folder):
    """
    Extracts RMSE data from specifications.txt files, organized by s2 and ls parameters.
    
    Parameters:
        base_folder (str): Path to the folder containing experiment results
        
    Returns:
        dict: Dictionary with structure {(s2, ls): {'SE_RMSE': [...], 'GSE_RMSE': [...]}}
    """
    import os
    import ast
    import numpy as np
    
    # Dictionary to store results organized by (s2, ls) pairs
    results_by_params = {}
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_folder):
        if "specifications.txt" in files:
            spec_path = os.path.join(root, "specifications.txt")
            
            try:
                with open(spec_path, 'r') as f:
                    lines = f.readlines()
                    
                    # Extract model parameters
                    model_line = [line for line in lines if line.startswith('model_params:')]
                    if not model_line:
                        continue
                    model_dict = ast.literal_eval(model_line[0].split('model_params:')[1].strip())
                    
                    # Extract performance parameters
                    perf_line = [line for line in lines if line.startswith('performance_params:')]
                    if not perf_line:
                        continue
                    perf_dict = ast.literal_eval(perf_line[0].split('performance_params:')[1].strip())
                    
                    # Get s2 and ls values
                    s2 = model_dict.get('MODEL_s2', None)
                    ls = model_dict.get('MODEL_ls', None)
                    
                    # Get RMSE values
                    se_rmse = perf_dict.get('SE_RMSE', None)
                    gse_rmse = perf_dict.get('GSE_RMSE', None)
                    
                    if s2 is not None and ls is not None and se_rmse is not None and gse_rmse is not None:
                        # Convert numpy types to Python types if needed
                        s2 = float(s2) if hasattr(s2, 'item') else float(s2)
                        ls = float(ls) if hasattr(ls, 'item') else float(ls)
                        se_rmse = float(se_rmse) if hasattr(se_rmse, 'item') else float(se_rmse)
                        gse_rmse = float(gse_rmse) if hasattr(gse_rmse, 'item') else float(gse_rmse)
                        
                        # Create key for this parameter combination
                        param_key = (s2, ls)
                        
                        # Initialize if this parameter combination doesn't exist
                        if param_key not in results_by_params:
                            results_by_params[param_key] = {'SE_RMSE': [], 'GSE_RMSE': []}
                        
                        # Add the RMSE values
                        results_by_params[param_key]['SE_RMSE'].append(se_rmse)
                        results_by_params[param_key]['GSE_RMSE'].append(gse_rmse)
                        
            except Exception as e:
                print(f"Error reading {spec_path}: {e}")
                continue
    
    return results_by_params

def calculate_rmse_statistics_by_parameters(base_folder):
    """
    Calculates mean and std dev of RMSE for each s2, ls parameter combination.
    
    Parameters:
        base_folder (str): Path to the folder containing experiment results
        
    Returns:
        dict: Dictionary with statistics for each parameter combination
    """
    results_by_params = extract_rmse_by_parameters(base_folder)
    
    statistics = {}
    
    for (s2, ls), rmse_data in results_by_params.items():
        se_rmses = rmse_data['SE_RMSE']
        gse_rmses = rmse_data['GSE_RMSE']
        
        if se_rmses and gse_rmses:
            # Calculate statistics
            se_mean = np.mean(se_rmses)
            se_std = np.std(se_rmses)
            gse_mean = np.mean(gse_rmses)
            gse_std = np.std(gse_rmses)
            
            # Calculate improvement statistics
            improvements = []
            for se, gse in zip(se_rmses, gse_rmses):
                if se != 0:
                    improvement = 100 * (se - gse) / se
                    improvements.append(improvement)
            
            improvement_mean = np.mean(improvements) if improvements else 0
            improvement_std = np.std(improvements) if improvements else 0
            
            statistics[(s2, ls)] = {
                'SE_RMSE_mean': se_mean,
                'SE_RMSE_std': se_std,
                'GSE_RMSE_mean': gse_mean,
                'GSE_RMSE_std': gse_std,
                'improvement_mean': improvement_mean,
                'improvement_std': improvement_std,
                'num_datasets': len(se_rmses)
            }
    
    return statistics

def print_rmse_statistics_by_parameters(base_folder):
    """
    Prints RMSE statistics organized by s2 and ls parameters.
    
    Parameters:
        base_folder (str): Path to the folder containing experiment results
    """
    statistics = calculate_rmse_statistics_by_parameters(base_folder)
    
    if not statistics:
        print("No results found in the specified folder.")
        return
    
    print("RMSE Statistics by Parameters (s2, ls):")
    print("=" * 80)
    
    # Sort by s2, then by ls for consistent output
    sorted_params = sorted(statistics.keys(), key=lambda x: (x[0], x[1]))
    
    for (s2, ls) in sorted_params:
        stats = statistics[(s2, ls)]
        print(f"\nParameters: s2={s2}, ls={ls} (n={stats['num_datasets']} datasets)")
        print(f"  SE_RMSE:     mean={stats['SE_RMSE_mean']:.4f}, std={stats['SE_RMSE_std']:.4f}")
        print(f"  GSE_RMSE:    mean={stats['GSE_RMSE_mean']:.4f}, std={stats['GSE_RMSE_std']:.4f}")
        print(f"  Improvement: mean={stats['improvement_mean']:.2f}%, std={stats['improvement_std']:.2f}%")
    
    # Calculate overall statistics across all parameter combinations
    all_se_rmses = []
    all_gse_rmses = []
    all_improvements = []
    
    for stats in statistics.values():
        all_se_rmses.extend([stats['SE_RMSE_mean']] * stats['num_datasets'])
        all_gse_rmses.extend([stats['GSE_RMSE_mean']] * stats['num_datasets'])
        all_improvements.extend([stats['improvement_mean']] * stats['num_datasets'])
    
    if all_se_rmses:
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS (across all parameter combinations):")
        print(f"  SE_RMSE:     mean={np.mean(all_se_rmses):.4f}, std={np.std(all_se_rmses):.4f}")
        print(f"  GSE_RMSE:    mean={np.mean(all_gse_rmses):.4f}, std={np.std(all_gse_rmses):.4f}")
        print(f"  Improvement: mean={np.mean(all_improvements):.2f}%, std={np.std(all_improvements):.2f}%")

# Example usage:
# print_rmse_statistics_by_parameters("Debugging/gSE_Params/s2Ls")
def get_rmse_data_for_visualization(base_folder):
    """
    Returns RMSE data organized for visualization (heatmaps, etc.).
    
    Parameters:
        base_folder (str): Path to the folder containing experiment results
        
    Returns:
        tuple: (s2_values, ls_values, se_rmse_matrix, gse_rmse_matrix, improvement_matrix)
    """
    statistics = calculate_rmse_statistics_by_parameters(base_folder)
    
    if not statistics:
        return None, None, None, None, None
    
    # Extract unique s2 and ls values
    s2_values = sorted(list(set(s2 for s2, ls in statistics.keys())))
    ls_values = sorted(list(set(ls for s2, ls in statistics.keys())))
    
    # Create matrices
    se_rmse_matrix = np.zeros((len(ls_values), len(s2_values)))
    gse_rmse_matrix = np.zeros((len(ls_values), len(s2_values)))
    improvement_matrix = np.zeros((len(ls_values), len(s2_values)))
    
    # Fill matrices
    for i, ls in enumerate(ls_values):
        for j, s2 in enumerate(s2_values):
            if (s2, ls) in statistics:
                stats = statistics[(s2, ls)]
                se_rmse_matrix[i, j] = stats['SE_RMSE_mean']
                gse_rmse_matrix[i, j] = stats['GSE_RMSE_mean']
                improvement_matrix[i, j] = stats['improvement_mean']
            else:
                # Fill with NaN if no data for this combination
                se_rmse_matrix[i, j] = np.nan
                gse_rmse_matrix[i, j] = np.nan
                improvement_matrix[i, j] = np.nan
    
    return s2_values, ls_values, se_rmse_matrix, gse_rmse_matrix, improvement_matrix

def save_rmse_to_txt(rmse_dict, filename, folder="Debugging"):
    """
    Save RMSE values from a dictionary to a text file.
    
    Parameters:
        rmse_dict (dict): Dictionary with model names as keys and RMSE values as values
        filename (str): Name of the output file
        folder (str): Folder to save the file in (default: 'Debugging')
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    
    with open(path, 'w') as f:
        f.write("RMSE Values by Model\n")
        f.write("=" * 30 + "\n\n")
        
        for model_name, rmse_value in rmse_dict.items():
            f.write(f"{model_name}: {rmse_value:.6f}\n")
        
        # Add summary statistics
        rmse_values = list(rmse_dict.values())
        if rmse_values:
            f.write(f"\nSummary Statistics:\n")
            f.write(f"Mean RMSE: {np.mean(rmse_values):.6f}\n")
            f.write(f"Std RMSE: {np.std(rmse_values):.6f}\n")
            f.write(f"Min RMSE: {np.min(rmse_values):.6f}\n")
            f.write(f"Max RMSE: {np.max(rmse_values):.6f}\n")
    
    print(f"RMSE values saved to {path}")

def find_convergence_time(base_folder, target_index, min_consecutive_steps=5):
    """
    Find the time step when goal predictions converge and stay at a specific index.
    
    Parameters:
        base_folder (str): Root folder to search (e.g., "Debugging/SE_MultTargets")
        target_index (int): The goal index to look for (0, 1, 2, or 3)
        min_consecutive_steps (int): Minimum number of consecutive steps at target_index to consider converged
        
    Returns:
        dict: Dictionary with folder paths as keys and convergence info as values
    """
    import os
    import numpy as np
    
    results = {}
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_folder):
        # Look for files that start with "best_goals"
        best_goals_files = [f for f in files if f.startswith("best_goals")]
        
        for file in best_goals_files:
            file_path = os.path.join(root, file)
            
            try:
                # Load the goal predictions
                goal_predictions = np.loadtxt(file_path)
                
                # If it's a 2D array, take the first column (goal index)
                if goal_predictions.ndim == 2:
                    goal_indices = goal_predictions[:, 0].astype(int)
                else:
                    goal_indices = goal_predictions.astype(int)
                
                # Find when the predictions become and stay at target_index
                convergence_time = None
                consecutive_count = 0
                
                for i, pred_idx in enumerate(goal_indices):
                    if pred_idx == target_index:
                        consecutive_count += 1
                        if consecutive_count >= min_consecutive_steps:
                            convergence_time = i - min_consecutive_steps + 1
                            break
                    else:
                        consecutive_count = 0
                
                # Store results
                relative_path = os.path.relpath(file_path, base_folder)
                results[relative_path] = {
                    'convergence_time': convergence_time,
                    'total_steps': len(goal_indices),
                    'final_prediction': goal_indices[-1] if len(goal_indices) > 0 else None,
                    'converged': convergence_time is not None
                }
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    return results

def print_convergence_summary(base_folder, target_index, min_consecutive_steps=5):
    """
    Print a summary of convergence times for a specific target index.
    
    Parameters:
        base_folder (str): Root folder to search
        target_index (int): The goal index to look for
        min_convergence_steps (int): Minimum consecutive steps to consider converged
    """
    results = find_convergence_time(base_folder, target_index, min_consecutive_steps)
    
    if not results:
        print(f"No 'best_goals' files found in {base_folder}")
        return
    
    print(f"Convergence Analysis for Target Index {target_index}")
    print("=" * 60)
    
    # Group by parameter combinations
    param_groups = {}
    for file_path, data in results.items():
        # Extract parameter info from path (e.g., "SG:0.5/Gvar:100/best_goals_...")
        path_parts = file_path.split(os.sep)
        if len(path_parts) >= 3:
            sg_part = path_parts[-3]  # e.g., "SG:0.5"
            gvar_part = path_parts[-2]  # e.g., "Gvar:100"
            param_key = f"{sg_part}_{gvar_part}"
            
            if param_key not in param_groups:
                param_groups[param_key] = []
            param_groups[param_key].append(data)
    
    # Print summary by parameter group
    for param_key, data_list in param_groups.items():
        print(f"\nParameter Group: {param_key}")
        print("-" * 40)
        
        converged_times = [d['convergence_time'] for d in data_list if d['converged']]
        not_converged = [d for d in data_list if not d['converged']]
        
        if converged_times:
            print(f"Converged: {len(converged_times)}/{len(data_list)} files")
            print(f"Average convergence time: {np.mean(converged_times):.1f} steps")
            print(f"Min convergence time: {np.min(converged_times)} steps")
            print(f"Max convergence time: {np.max(converged_times)} steps")
            print(f"Std convergence time: {np.std(converged_times):.1f} steps")
        else:
            print(f"Converged: 0/{len(data_list)} files")
        
        if not_converged:
            print(f"Not converged: {len(not_converged)} files")
            final_predictions = [d['final_prediction'] for d in not_converged]
            print(f"Final predictions for non-converged: {final_predictions}")
    
    # Overall summary
    all_converged_times = [d['convergence_time'] for d in results.values() if d['converged']]
    total_files = len(results)
    converged_files = len(all_converged_times)
    
    print(f"\nOverall Summary:")
    print(f"Total files analyzed: {total_files}")
    print(f"Files converged to index {target_index}: {converged_files}")
    print(f"Convergence rate: {converged_files/total_files*100:.1f}%")
    
    if all_converged_times:
        print(f"Overall average convergence time: {np.mean(all_converged_times):.1f} steps")
        print(f"Overall convergence time range: {np.min(all_converged_times)} - {np.max(all_converged_times)} steps")
