"""
test if shd runs
"""
import numpy as np
from causalspyne.ancestral_shd import structural_hamming_distance


def test_shd():
    """
    test shd runs
    """
    target_graph = np.array([[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0]])

    predicted_graph = np.array([[0, 0, 1],
                                [0, 0, 1],
                                [0, 0, 0]])

    shd = structural_hamming_distance(target_graph, predicted_graph)
    print(f"Structural Hamming Distance: {shd}")

    # With double_for_anticausal=False
    shd_single = structural_hamming_distance(target_graph, predicted_graph,
                                             double_for_anticausal=False)
    print(f"SHD (single count for reversed edges): {shd_single}")
