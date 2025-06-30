import os
import numpy as np

def make_groundtruth(filename):
    data = np.loadtxt(filename, comments="#")

    tx_ty = data[:, [1,2]]

    #tx_ty_list = [np.array([tx, ty]) for tx, ty in tx_ty]

    return tx_ty

def pretty_print_matrix(matrix, name, file):
    file.write(f"{name} (shape {matrix.shape}):\n")
    for row in matrix:
        file.write("  " + "  ".join(f"{val:8.4f}" for val in row) + "\n")
    file.write("\n")

def save_vector_arrays_txt(arr1, arr2, filename, label1, label2):
    """
    Save two arrays of vectors (shape: (N, d)) to a txt file in Debugging/ with readable formatting.
    """
    os.makedirs('Debugging', exist_ok=True)
    path = os.path.join('Debugging', filename)
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

def save_matrix_arrays_txt(arr1, arr2, filename, label1, label2):
    """
    Save two arrays of matrices (shape: (N, d, d)) to a txt file in Debugging/ with readable formatting, using pretty_print_matrix for each matrix.
    """
    os.makedirs('Debugging', exist_ok=True)
    path = os.path.join('Debugging', filename)
    with open(path, 'w') as f:
        for k in range(min(len(arr1), len(arr2))):
            f.write(f"Step {k}:\n")
            pretty_print_matrix(arr1[k], f"{label1} Example {k+1}", f)
            pretty_print_matrix(arr2[k], f"{label2} Example {k+1}", f)
            f.write("\n" + "="*50 + "\n\n")

def save_state_comparison_txt(full_state_iSE, full_state_goal, filename):
    """
    Save a detailed state comparison between iSE and goal model state arrays to Debugging/filename.
    """
    os.makedirs('Debugging', exist_ok=True)
    path = os.path.join('Debugging', filename)
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