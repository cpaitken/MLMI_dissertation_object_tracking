import os
import numpy as np
import matplotlib.pyplot as plt
import re
import ast

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
    "print_rmse_summary"
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

def save_tracking_plot(groundtruth, noisy_data, X, G, XN, modelName1, modelName2, filename, folder="Debugging"):
    """
    Plot and save the tracking results to Debugging/Plots/filename (PNG).
    """
    os.makedirs(os.path.join(folder), exist_ok=True)
    plt.figure()
    plt.plot(groundtruth[:,0], groundtruth[:,1], label='Truth')
    plt.scatter(*zip(*noisy_data), alpha=0.3, label='Noisy obs')
    plt.plot(X[:,0], X[:,1], label=modelName1, color='green')
    plt.plot(G[:,0], G[:,1], label='Inferred goal', color='darkred')
    plt.plot(XN[:,0], XN[:,1], '--', label=modelName2, color='limegreen')
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
    """
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
                    if se_rmse != 0:
                        improvement = 100 * (se_rmse - gse_rmse) / se_rmse
                        improvements.append(improvement)
        if se_rmses:
            print(f"Category: {category}")
            print(f"  SE_RMSE: mean={np.mean(se_rmses):.4f}, std={np.std(se_rmses):.4f}")
            print(f"  GSE_RMSE: mean={np.mean(gse_rmses):.4f}, std={np.std(gse_rmses):.4f}")
            print(f"  GSE % improvement over SE: mean={np.mean(improvements):.2f}%, std={np.std(improvements):.2f}%\n")
        else:
            print(f"Category: {category} (no valid results found)")

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