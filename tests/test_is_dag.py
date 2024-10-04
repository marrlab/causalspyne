"""
toplogical sort
"""

import numpy as np

from causalspyne.is_dag import is_dag


def test_is_dag0():
    """
    adj must be lower triangular
    """
    adj_matrix = np.array(
        [
            [0, 0],  #
            [1, 0],  # 1->0
        ]
    )

    is_dag(adj_matrix)


def test_is_dag():
    """
    adj must be lower triangular
    """
    adj_matrix = np.array(
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]  # 1->0  # 2->1  # 3->2
    )

    is_dag(adj_matrix)


def test_is_dag2():
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
    is_dag(adj_matrix)
