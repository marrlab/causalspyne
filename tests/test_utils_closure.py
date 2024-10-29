"""
check if transitive closure correct for ancestor relationship determination
"""
import numpy as np
from causalspyne.utils_closure import ancestor_matrix


def test_utils_transitive_closure():
    """
    check if transitive closure correct
    """
    # Example usage note this adj matrix is opposite of our convention
    # A->B, B->C, C->D
    #     A, B, C, D
    adj_matrix = np.array(
        [[0, 1, 0, 0],     # A row as parent
         [0, 0, 1, 0],     # B row as parent
         [0, 0, 0, 1],     # C row as parent
         [0, 0, 0, 0]])    # D row as parent
    pred_ancestor_mat = ancestor_matrix(adj_matrix)
    print("pred Ancestor matrix:")
    print(pred_ancestor_mat.astype(int))
    # A->C(new)
    # A->D(new)
    # B->C(new)
    #  A B C D
    # [[
    #  0 1 1 1]   # A row as parent
    # [0 0 1 1]   # B row as parent
    # [0 0 0 1]   # C row as parent
    # [0 0 0 0]]  # D row as parent

    #     A, B, C, D
    ground_truth_ancestor_matrix = np.array(
        [[0, 1, 1, 1],     # A row as parent
         [0, 0, 1, 1],     # B row as parent
         [0, 0, 0, 1],     # C row as parent
         [0, 0, 0, 0]])    # D row as parent
    assert (ground_truth_ancestor_matrix == pred_ancestor_mat).all()
