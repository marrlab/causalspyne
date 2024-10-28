"""
test data and DAG subview
"""

from numpy.random import default_rng
import numpy as np
from causalspyne.dag2ancestral import DAG2Ancestral


def test_data_dag_subview():
    """
    test linear gaussian data gen
    """
    adj_matrix = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    # 0->1, 1->2, 2->3
    # if we hide 1
    # 0->2, 2->3
    # submatrix
    # 0, 2, 3
    sub_matrix = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])
    # correct ancestral graph
    ancestor_graph_matrix = sub_matrix
    obj = DAG2Ancestral(adj_matrix)
    pred_ancestral_graph = obj.run([1])
    assert (ancestor_graph_matrix == pred_ancestral_graph).all()
