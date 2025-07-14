import seaborn as sns
import pandas as pd
from dataFunctions import get_rmse_data_for_visualization
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm

def create_annotation_matrix(matrix, show_min_max=True):
    """
    Create annotation matrix showing only min/max values or all values.
    
    Parameters:
        matrix: numpy array of values
        show_min_max: if True, only show min and max values, otherwise show all
    
    Returns:
        annotation_matrix: matrix with strings for annotations
    """
    annotation_matrix = np.full(matrix.shape, '', dtype=object)
    
    if show_min_max:
        # Find min and max values (ignoring NaN)
        valid_values = matrix[~np.isnan(matrix)]
        if len(valid_values) > 0:
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            
            # Mark min and max positions
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if not np.isnan(matrix[i, j]):
                        if matrix[i, j] == min_val:
                            annotation_matrix[i, j] = f'{matrix[i, j]:.3f} (min)'
                        elif matrix[i, j] == max_val:
                            annotation_matrix[i, j] = f'{matrix[i, j]:.3f} (max)'
    else:
        # Show all values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    annotation_matrix[i, j] = f'{matrix[i, j]:.3f}'
    
    return annotation_matrix

def create_annotation_matrix_improvement(matrix):
    """
    For improvement matrix: show best (highest) and worst (lowest) improvements.
    """
    annotation_matrix = np.full(matrix.shape, '', dtype=object)
    
    valid_values = matrix[~np.isnan(matrix)]
    if len(valid_values) > 0:
        best_improvement = np.max(valid_values)
        worst_improvement = np.min(valid_values)
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    if matrix[i, j] == best_improvement:
                        annotation_matrix[i, j] = f'{matrix[i, j]:.1f}% (best)'
                    elif matrix[i, j] == worst_improvement:
                        annotation_matrix[i, j] = f'{matrix[i, j]:.1f}% (worst)'
    
    return annotation_matrix

def create_annotation_matrix_simple(matrix):
    """
    Create annotation matrix showing only min and max values without labels.
    """
    annotation_matrix = np.full(matrix.shape, '', dtype=object)
    valid_values = matrix[~np.isnan(matrix)]
    if len(valid_values) > 0:
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    if matrix[i, j] == min_val or matrix[i, j] == max_val:
                        annotation_matrix[i, j] = f'{matrix[i, j]:.3f}'
    return annotation_matrix

s2_vals, ls_vals, se_matrix, gse_matrix, imp_matrix = get_rmse_data_for_visualization("Debugging/gSE_Params/s2Ls")

# SE Model RMSE
se_df = pd.DataFrame(se_matrix, index=ls_vals, columns=s2_vals)
se_annot = create_annotation_matrix_simple(se_matrix)
plt.figure(figsize=(6, 6))
sns.heatmap(se_df, annot=se_annot, fmt='', cmap='Reds', square=True)
plt.title('SE Model RMSE')
plt.xlabel('s2')
plt.ylabel('ls')
plt.tight_layout()
plt.show()

# GSE Model RMSE
gse_df = pd.DataFrame(gse_matrix, index=ls_vals, columns=s2_vals)
gse_annot = create_annotation_matrix_simple(gse_matrix)
plt.figure(figsize=(6, 6))
sns.heatmap(gse_df, annot=gse_annot, fmt='', cmap='Blues', square=True)
plt.title('GSE Model RMSE')
plt.xlabel('s2')
plt.ylabel('ls')
plt.tight_layout()
plt.show()

# Improvement: white for <=0, green for >0, darker green for higher improvement
imp_df = pd.DataFrame(imp_matrix, index=ls_vals, columns=s2_vals)
imp_annot = create_annotation_matrix_simple(imp_matrix)

# Create a custom colormap: white for 0 or less, green for positive
max_improvement = np.nanmax(imp_matrix)
# We'll use 256 colors, first color is white, rest are from Greens
n_colors = 256
colors = plt.get_cmap('Greens', n_colors)
# Make a new colormap where the first color is white, rest are Greens
newcolors = colors(np.linspace(0, 1, n_colors))
newcolors[0, :] = [1, 1, 1, 1]  # RGBA for white
custom_green = ListedColormap(newcolors)

# Normalize: all values <=0 map to 0 (white), >0 map to (0, max_improvement)
class ZeroWhiteNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False):
        super().__init__(vmin, vmax, clip)
    def __call__(self, value, clip=None):
        value = np.array(value)
        normed = np.zeros_like(value, dtype=float)
        # All values <= 0 are 0 (white)
        normed[value > 0] = (value[value > 0] - 0) / (self.vmax - 0) if self.vmax > 0 else 0
        return normed

norm = ZeroWhiteNormalize(vmin=0, vmax=max_improvement)

plt.figure(figsize=(6, 6))
sns.heatmap(imp_df, annot=imp_annot, fmt='', cmap=custom_green, square=True, norm=norm, cbar_kws={'label': 'Improvement (%)'})
plt.title('GSE Improvement over SE (%)')
plt.xlabel('s2')
plt.ylabel('ls')
plt.tight_layout()
plt.show()