"""
"""

from causallearn.utils.DAG2PAG import dag2pag
from causallearn.graph.Dag import Dag
import numpy as np

import numpy as np


def ancestral_acc(true_dag, true_hidden_nodes, pred_order):
    """

    Parameters:
    true_dag (numpy.ndarray): Target graph adjacency matrix
    true_hidden_nodes (list): list of nodes in DAG to hide
    Returns:
    int:
    """
    n = len(true_dag) - len(true_hidden_nodes)
    if (n, n) != prediction.shape:
        raise ValueError("Graphs must have the same number of nodes")

    cl_dag = Dag(range(len(true_dag)))
    for ch, pa in np.argwhere(true_dag):
        cl_dag.add_directed_edge(pa, ch)
    true_pag = dag2pag(cl_dag, true_hidden_nodes)

    total_shd = np.sum(true_pag.graph != prediction)

    return total_shd / (n**2 - n)
