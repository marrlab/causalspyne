"""
"""

# class ancestral_shd():
    #def __init__(mat_gt_ancestral, mat_fci):
    #    self.mat_gt_ancestral = mat_gt_ancestral


import numpy as np

def structural_hamming_distance(target, prediction, double_for_anticausal=True):
    """
    Compute the Structural Hamming Distance between two graphs.

    Parameters:
    target (numpy.ndarray): Target graph adjacency matrix
    prediction (numpy.ndarray): Predicted graph adjacency matrix
    double_for_anticausal (bool): Count badly oriented edges as two mistakes

    Returns:
    int: Structural Hamming Distance
    """
    if target.shape != prediction.shape:
        raise ValueError("Graphs must have the same number of nodes")

    # Ensure the matrices are binary
    target = (target != 0).astype(int)
    prediction = (prediction != 0).astype(int)

    # Compute the difference
    diff = np.abs(target - prediction)

    if double_for_anticausal:
        # Count reversed edges twice
        return np.sum(diff)
    # Count reversed edges once
    return np.sum(np.maximum(diff, diff.T)) // 2
