"""
test if shd runs
"""

import numpy as np
from causalspyne.ancestral_shd import structural_hamming_distance


def test_shd():
    """
    test shd runs
    """
    target_graph = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    predicted_graph = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    shd = structural_hamming_distance(target_graph, [], predicted_graph)
    print(f"Structural Hamming Distance: {shd}")


def test_shd_with_hidden_node():
    target_graph = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    predicted_graph = np.array([[0, 1], [1, 0]])

    shd = structural_hamming_distance(target_graph, [0], predicted_graph)

    assert np.isfinite(shd)
    assert 0 <= shd <= 1
