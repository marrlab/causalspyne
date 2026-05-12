"""
"""

from causallearn.utils.DAG2PAG import dag2pag
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
import numpy as np

# class ancestral_shd():
# def __init__(mat_gt_ancestral, mat_fci):
#    self.mat_gt_ancestral = mat_gt_ancestral


import numpy as np


def structural_hamming_distance(true_dag, true_hidden_nodes, prediction):
    """
    Compute the standardized structural Hamming distance between two graphs.

    Parameters:
    true_dag (numpy.ndarray): Target graph adjacency matrix
    true_hidden_nodes (list): list of nodes in DAG to hide
    prediction (numpy.ndarray): Predicted graph adjacency matrix  (pag.graph
    from causal-learn)
    double_for_anticausal (bool): Count badly oriented edges as two mistakes

    Returns:
    int: Structural Hamming Distance
    """
    n = len(true_dag) - len(true_hidden_nodes)
    if (n, n) != prediction.shape:
        raise ValueError("Graphs must have the same number of nodes")

    nodes = [GraphNode(f"X{i}") for i in range(len(true_dag))]
    hidden_nodes = [nodes[int(ind)] for ind in true_hidden_nodes]

    cl_dag = Dag(nodes)
    for ch, pa in np.argwhere(true_dag):
        cl_dag.add_directed_edge(nodes[pa], nodes[ch])
    true_pag = dag2pag(cl_dag, hidden_nodes)

    total_shd = np.sum(true_pag.graph != prediction)

    return total_shd / (n**2 - n)
