# Import packages
import numpy as np

def norm_and_invert(similarity_matrix):
    """Post processing of the matrices. Has numpy matrix as input argument and normalizes and inverts all the values."""

    # Divide whole matrix by its maximum value

    max_value = np.max(similarity_matrix)
    new_mat = similarity_matrix/max_value

    # Invert values in matrix

    inv_mat = 1/new_mat

    filtered_data = []

    # Add everything but diagonal to filtered data because diagonal contains infinity symbols

    for row in range(inv_mat.shape[0]):
        for column in range(inv_mat.shape[1]):
            if row != column:
                filtered_data.append(inv_mat[row][column])      # Create new matrix with filtered data

    # Normalize data again by dividing by the maximum value

    norm_mat = np.round(inv_mat/np.max(filtered_data), decimals=2)

    return norm_mat
