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
    #     0, 1, 2, 3 (column index)
    #     A, B, C, D (as parent)
    adj_matrix = np.array(
        [[0, 0, 0, 0],     # row index 0, A as child
         [1, 0, 0, 0],     # row index 1, B as child
         [0, 1, 0, 0],     # row index 2, C as child
         [0, 0, 1, 0]]     # row index 3, D as child
    )

    # - entry(1,0): entry(B, A), A->B
    # - entry(2,1): etnry(C, B), B->C
    # - entry(3,2): C->D
    #

    # if we hide B with index 1
    # 0->2 or A->C  (* new)
    # 2->3 or C->D  (edge left intact)

    # the full matrix
    #      0, 1, 2, 3 (column index, parent)
    #      A, B, C, D
    #    [[0, 0, 0, 0],   # row index 0, A as child
    #     [1, 0, 0, 0],   # row index 1, B as child
    #     [1, 1, 0, 0],   # row index 2, C as child
    #     [0, 0, 1, 0]])  # row index 3, D as child
    # new edge is (2,0) or (C, A) correponding to A->C

    # delete index 1 row
    #      0, 1, 2, 3 (column index, parent)
    #      A, B, C, D
    #    [[0, 0, 0, 0],   # row index 0, A as child
    #     [1, 1, 0, 0],   # row index 2, C as child
    #     [0, 0, 1, 0]])  # row index 3, D as child

    # delete index 1 column
    #      0, 2, 3
    #      A, C, D (parent)
    #    [[0, 0, 0],   # row index 0, A as child
    #     [1, 0, 0],   # row index 2, C as child
    #     [0, 1, 0]])  # row index 3, D as child

    sub_matrix = np.array(
        [[0, 0, 0],
         [1, 0, 0],
         [0, 1, 0]])

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
    #        0, 1, 2, 3, 4, 5, 6, 7 (column ind, parent)
    #        A, B, H, C, D, E, F, G (as parent)
    adj_matrix = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],  # row ind 0, A as child
            [1, 0, 0, 0, 0, 0, 0, 0],  # row ind 1, B as child
            [0, 1, 0, 0, 0, 0, 0, 0],  # row ind 2, H as child
            [0, 0, 1, 0, 0, 0, 0, 0],  # row ind 3, C as child
            [0, 0, 0, 1, 0, 0, 0, 0],  # row ind 4, D as child
            [0, 0, 1, 0, 1, 0, 0, 0],  # row ind 5, E as child
            [0, 0, 1, 0, 0, 0, 0, 0],  # row ind 6, F as child
            [0, 0, 0, 0, 0, 0, 1, 0],  # row ind 7, G as child
        ]
    )

    # Updated Graph after hiding H:
    # A -> B
    # B -> {C, E, F} *new
    # C -> D -> E
    # C -> E *new
    # F -> G
    # C <-> F <-> E *new
    #         0, 1, 2, 3, 4, 5, 6, 7 (column ind, parent)
    #         A, B, H, C, D, E, F, G (as parent)

    #        [0, 0, 0, 0, 0, 0, 0, 0],  # row ind 0, A as child
    #        [1, 0, 0, 0, 0, 0, 0, 0],  # row ind 1, B as child
    #        [0, 1, 0, 0, 0, 0, 0, 0],  # row ind 2, H as child
    #        [0,(1),1, 0, 0, 0,{1},0],  # row ind 3, C as child
    #        [0, 0, 0, 1, 0, 0, 0, 0],  # row ind 4, D as child
    #        [0,(1),1,(1),1, 0,{1},0],  # row ind 5, E as child
    #        [0,(1),1,{1},0,{1},0, 0],  # row ind 6, F as child
    #        [0, 0, 0, 0, 0, 0, 1, 0],  # row ind 7, G as child

    # delete H row

    #         0, 1, 2, 3, 4, 5, 6, 7 (column ind, parent)
    #         A, B, H, C, D, E, F, G (as parent)

    #        [0, 0, 0, 0, 0, 0, 0, 0],  # row ind 0, A as child
    #        [1, 0, 0, 0, 0, 0, 0, 0],  # row ind 1, B as child
    #        [0,(1),1, 0, 0, 0,{1},0],  # row ind 3, C as child
    #        [0, 0, 0, 1, 0, 0, 0, 0],  # row ind 4, D as child
    #        [0,(1),1,(1),1, 0,{1},0],  # row ind 5, E as child
    #        [0,(1),1,{1},0,{1},0, 0],  # row ind 6, F as child
    #        [0, 0, 0, 0, 0, 0, 1, 0],  # row ind 7, G as child

    # delete H column
    #         0, 1, , 3, 4, 5, 6, 7 (column ind, parent)
    #         A, B, , C, D, E, F, G (as parent)

    #        [0, 0, , 0, 0, 0, 0, 0],  # row ind 0, A as child
    #        [1, 0, , 0, 0, 0, 0, 0],  # row ind 1, B as child
    #        [0,(1),, 0, 0, 0,{1},0],  # row ind 3, C as child
    #        [0, 0, , 1, 0, 0, 0, 0],  # row ind 4, D as child
    #        [0,(1),,(1),1, 0,{1},0],  # row ind 5, E as child
    #        [0,(1),,{1},0,{1},0, 0],  # row ind 6, F as child
    #        [0, 0, , 0, 0, 0, 1, 0],  # row ind 7, G as child

    #        submatrix
    #        A, B, C, D, E, F, G (parent)
    ancestral_graph = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],  # A as child
            [1, 0, 0, 0, 0, 0, 0],  # B as child
            [0, 1, 0, 0, 0, 1, 0],  # C as child
            [0, 0, 1, 0, 0, 0, 0],  # D as child
            [0, 1, 1, 1, 0, 1, 0],  # E as child
            [0, 1, 1, 0, 1, 0, 0],  # F as child
            [0, 0, 0, 0, 0, 1, 0],  # G as child
        ]
    )

    obj = DAG2Ancestral(adj_matrix)
    pred_ancestral_graph = obj.run([2])  # hide H
    assert (ancestral_graph == pred_ancestral_graph).all()
