"""
test generation of true ancestral graph from ground truth DAG
"""

import numpy as np
from causalspyne.dag2ancestral import DAG2Ancestral


def test_DAG2Ancestral_path():
    """
    hide one node in a path
    """
    adj_matrix = np.array(
        [[0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0]])

    # - entry(1,0): 0->1
    # - entry(2,1): 1->2
    # - entry(3,2): 2->3

    # if we hide 1
    # 0->2, 2->3
    # submatrix
    # 0, 2, 3
    sub_matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    # correct ancestral graph
    ancestor_graph_matrix = sub_matrix
    obj = DAG2Ancestral(adj_matrix)
    pred_ancestral_graph = obj.run([1])
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
    #        G, F, E, D, C, H, B, A
    adj_matrix = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],  # G
            [1, 0, 0, 0, 0, 0, 0, 0],  # F
            [0, 0, 0, 0, 0, 0, 0, 0],  # E
            [0, 0, 1, 0, 0, 0, 0, 0],  # D
            [0, 0, 0, 1, 0, 0, 0, 0],  # C
            [0, 1, 1, 0, 1, 0, 0, 0],  # H
            [0, 0, 0, 0, 0, 1, 0, 0],  # B
            [0, 0, 0, 0, 0, 0, 1, 0],  # A
        ]
    )

    # Updated Graph after hiding H:
    # A -> B
    # B -> {C, E, F} *new
    # C -> D -> E
    # C -> E *new
    # F -> G
    # C <-> F <-> E *new

    #        G, F, E, D, C, B, A
    ancestral_graph = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],  # G
            [1, 0, 1, 0, 1, 0, 0],  # F
            [0, 1, 0, 0, 0, 0, 0],  # E
            [0, 0, 1, 0, 0, 0, 0],  # D
            [0, 1, 1, 1, 0, 0, 0],  # C
            [0, 1, 1, 0, 1, 0, 0],  # B
            [0, 0, 0, 0, 0, 1, 0],  # A
        ]
    )

    obj = DAG2Ancestral(adj_matrix)
    pred_ancestral_graph = obj.run([5])  # hide H
    assert (ancestral_graph == pred_ancestral_graph).all()
