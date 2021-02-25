#%% Import packages
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

def make_heatmap(matrix, data_list, output_path=None, auc = 'no'):
    """Convert a numpy matrix to a heatmap and save the figure"""

    # Define dataset list for similarity matrices (5 by 5) and AUC matrix (3 by 5)

    datasets_list = data_list
    target_data = ['chest_xray', 'ISIC2018', 'pcam']

    # Specify dataset depending on AUC or similarity and change values on diagonal to Nan values

    if auc == 'no':
        x_axis_labels = datasets_list
        y_axis_labels = datasets_list

        for column in range(matrix.shape[1]):
            for row in range(matrix.shape[0]):
                if column == row:
                    matrix[row][column] = np.nan

    elif auc == 'yes':
        x_axis_labels = datasets_list
        y_axis_labels = target_data

        for column in range(matrix.shape[1]):
            for row in range(matrix.shape[0]):
                if column == 0 and row == 0:
                    matrix[row][column] = np.nan
                elif column == 2 and row == 1:
                    matrix[row][column] = np.nan
                elif column == 4 and row == 2:
                    matrix[row][column] = np.nan

    # Create heatmap

    heat_map = sb.heatmap(matrix, cmap='Greens', xticklabels=x_axis_labels, yticklabels = y_axis_labels,
                          annot=True, annot_kws={'size':16}, cbar_kws={'label': 'Colorbar', 'orientation': 'horizontal'}, fmt='.3g')

    plt.show()

    # Save figure to specified location
    if (output_path != None):
        figure = heat_map.get_figure()
        figure.savefig(output_path)

    return
