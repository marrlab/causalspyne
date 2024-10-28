import numpy as np
from causalspyne.utils_closure import ancestor_matrix_transpose


def test_utils_transitive_closure():
    # Example usage note this adj matrix is opposite of our convention
    adj_matrix = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    ancestor_mat = ancestor_matrix_transpose(adj_matrix)
    print("Ancestor matrix:")
    print(ancestor_mat.astype(int))
