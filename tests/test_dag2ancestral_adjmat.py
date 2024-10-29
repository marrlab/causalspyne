"""
test generation of true ancestral graph from ground truth DAG
"""

import numpy as np
from causalspyne.dag2ancestral import DAG2Ancestral


def test_DAG2Ancestral_path():
    """
    hide one node in a path
    """
    # lower triangular requires reverse top order
    #     0, 1, 2, 3
    #     A, B, C, D
    adj_matrix = np.array(
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]  # A  # B  # C
    )  # D

    # - entry(1,0): entry(B, A), A->B
    # - entry(2,1): B->C
    # - entry(3,2): A->B
    #

    # if we hide 1
    # 0->2 * new
    # 2->3 intact
    # the full matrix

    #      3, 2, 1, 0

    #    [[0, 0, 0, 0],   # 3
    #     [1, 0, 0, 0],   # 2
    #     [0, 1, 0, 0],   # 1
    #     [0, 1, 1, 0]])  # 0
    # delete index 2 row
    #    [[0, 0, 0, 0],   # 3
    #     [1, 0, 0, 0],   # 2
    #     [0, 1, 1, 0]])  # 0

    # delete index 2 column
    #    [[0, 0, 0],   # 3
    #     [1, 0, 0],   # 2
    #     [0, 1, 0]])  # 0

    # submatrix
    #     3, 2, 0
    sub_matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # 3  # 2  # 0
    # Y=Bu
    # correct ancestral graph
    ancestor_graph_matrix = sub_matrix
    obj = DAG2Ancestral(adj_matrix)
    pred_ancestral_graph = obj.run([2])
    assert (ancestor_graph_matrix == pred_ancestral_graph).all()


def test_DAG2Ancestral_complicated():
    """
    hide a node with parents and multiple children, resulting in both
    kinds of directed edges as well as bidirected edges

    Original Graph:

     A -> B -> H -> {C, E, F}
     C -> D -> E
     F -> G

    """
    #        0, 1, 2, 3, 4, 5, 6, 7
    #        A, B, H, C, D, E, F, G
    adj_matrix = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],  # A
            [1, 0, 0, 0, 0, 0, 0, 0],  # B
            [0, 1, 0, 0, 0, 0, 0, 0],  # H
            [0, 0, 1, 0, 0, 0, 0, 0],  # C
            [0, 0, 0, 1, 0, 0, 0, 0],  # D
            [0, 0, 1, 0, 1, 0, 0, 0],  # E
            [0, 0, 1, 0, 0, 0, 0, 0],  # F
            [0, 0, 0, 0, 0, 0, 1, 0],  # G
        ]
    )

    # Updated Graph after hiding H:
    # A -> B
    # B -> {C, E, F} *new
    # C -> D -> E
    # C -> E *new
    # F -> G
    # C <-> F <-> E *new

    #        A, B, C, D, E, F, G
    ancestral_graph = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],  # A
            [1, 0, 0, 0, 0, 0, 0],  # B
            [0, 1, 0, 0, 0, 1, 0],  # C
            [0, 0, 1, 0, 0, 0, 0],  # D
            [0, 1, 1, 1, 0, 1, 0],  # E
            [0, 1, 1, 0, 1, 0, 0],  # F
            [0, 0, 0, 0, 0, 1, 0],  # G
        ]
    )

    obj = DAG2Ancestral(adj_matrix)
    pred_ancestral_graph = obj.run([2])  # hide H
    assert (ancestral_graph == pred_ancestral_graph).all()
