"""
toplogical sort on adjacency matrix
"""


import numpy as np
from causalspyne.is_dag import is_dag


def is_binary_matrix(matrix):
    """
    check if matrix is binary
    """
    # Convert the input to a NumPy array if it isn't already
    arr = np.array(matrix)

    # Check if all elements are either 0 or 1
    return np.all((arr == 0) | (arr == 1))


def topological_sort(binary_adj_mat):
    """
    first identify all source nodes

    if binary_adj_mat[node_arrow_head][node_arrow_tail] != 0, then there is
    edge from node_arrow_tail to node_arrow_head since by default we assume
    a lower triangular matrix where the first row should be source of the graph
    since no other nodes point into it.
    """
    if not is_dag(binary_adj_mat):
        raise RuntimeError("not a DAG!")
    if not is_binary_matrix(binary_adj_mat):
        raise RuntimeError("input matrix must only have 1, 0 for counting!")
    num_nodes = len(binary_adj_mat)
    # np.sum([[0, 1], [0, 5]], axis=0)
    # array([0, 6])
    arr_node_in_degree_volatile = np.sum(binary_adj_mat, axis=1)
    # list_queue_src_node_inds initially only contains all source nodes
    list_queue_src_node_inds = [i for i in range(num_nodes)
                                if arr_node_in_degree_volatile[i] == 0]
    if not list_queue_src_node_inds:
        raise RuntimeError("no source nodes!")
    list_sorted_node_inds = []

    while list_queue_src_node_inds:
        node_src = list_queue_src_node_inds.pop(0)
        list_sorted_node_inds.append(node_src)
        for neighbor in range(num_nodes):
            if binary_adj_mat[neighbor][node_src] != 0:
                # arrow: node_src -> neighbor
                arr_node_in_degree_volatile[neighbor] -= 1
                if arr_node_in_degree_volatile[neighbor] == 0:
                    list_queue_src_node_inds.append(neighbor)

    if len(list_sorted_node_inds) != num_nodes:
        raise ValueError("sorted nodes not completed!: " + \
                         str(list_sorted_node_inds) + \
                         str(arr_node_in_degree_volatile))

    return list_sorted_node_inds
