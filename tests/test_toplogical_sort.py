"""
toplogical sort
"""

import numpy as np

from causalspyne.utils_topological_sort import topological_sort


def test_topological_sort_linear():
    """
    adj must be lower triangular
    """
    adj_matrix = np.array(
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]  # 1->0  # 2->1  # 3->2
    )

    order = topological_sort(adj_matrix)
    assert order == [3, 2, 1, 0]


def test_topological_sort():
    """
    adj must be lower triangular
    """
    adj_matrix = np.array(
        [
            # 0 1 2 3
            [0, 1, 1, 0],  # 0: 1->0, 2->0
            [0, 0, 1, 0],  # 1: 2->1
            [0, 0, 0, 1],  # 2: 3->2
            [0, 0, 0, 0],  # 3: all zero row is source
        ]
    )

    order = topological_sort(adj_matrix)
    assert 3 == order[0]
