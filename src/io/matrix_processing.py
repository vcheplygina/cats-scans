#%%
import numpy as np

def norm_and_invert(similarity_matrix):
    max_value = np.max(similarity_matrix)
    new_mat = similarity_matrix/max_value

    inv_mat = 1/new_mat

    filtered_data = []

    for row in range(inv_mat.shape[0]):
        for column in range(inv_mat.shape[1]):
            if row != column:
                filtered_data.append(inv_mat[row][column])

    norm_mat = np.round(inv_mat/np.max(filtered_data), decimals=2)

    return norm_mat