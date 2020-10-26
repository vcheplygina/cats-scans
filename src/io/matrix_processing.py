#%%
import numpy as np

def row_normalization(similarity_matrix):
    max_value = np.max(similarity_matrix)
    new_mat = similarity_matrix/max_value
    # for row_index in range(len(similarity_matrix)):
    #     max_row_value = np.max(similarity_matrix[row_index])
    #     similarity_matrix[row_index] = similarity_matrix[row_index]/max_row_value
    norm_mat = np.round(new_mat, decimals=2)

    return norm_mat