import os
import numpy as np
import matplotlib.pyplot as plt

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
    "save_state_array_txt"
]

def make_groundtruth(filename, UZH=False):
    data = np.loadtxt(filename, comments="#")

    if UZH:
        tx_ty = data[:, [1,2]]
    else:
        tx_ty = data[:, [0,1]]

    #tx_ty_list = [np.array([tx, ty]) for tx, ty in tx_ty]

    return tx_ty

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
    os.makedirs(os.path.join(folder, 'Plots'), exist_ok=True)
    plt.figure()
    plt.plot(groundtruth[:,0], groundtruth[:,1], label='Truth')
    plt.scatter(*zip(*noisy_data), alpha=0.3, label='Noisy obs')
    plt.plot(X[:,0], X[:,1], label=modelName1, color='green')
    plt.plot(G[:,0], G[:,1], label='Inferred goal', color='darkred')
    plt.plot(XN[:,0], XN[:,1], '--', label=modelName2, color='limegreen')
    plt.legend()
    plt.savefig(os.path.join(folder, 'Plots', filename), bbox_inches='tight')
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