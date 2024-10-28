import numpy as np


def transitive_closure(adj_matrix):
    """
    Floyd-Warshall algorithm
    """
    num_vertex = len(adj_matrix)
    closure = np.array(adj_matrix, dtype=bool)

    for k in range(num_vertex):  # intermiediate connection
        for i in range(num_vertex):
            for j in range(num_vertex):
                # as long as one of the three cases result in connection, then
                # i, j should be connected
                closure[i][j] = closure[i][j] or \
                    (closure[i][k] and closure[k][j])
                # i-k, k-j implies i-j
    return closure


def ancestor_matrix_transpose(adj_matrix):
    num_vertex = len(adj_matrix)
    closure = transitive_closure(adj_matrix)
    ancestor = np.zeros((num_vertex, num_vertex), dtype=bool)

    for i in range(num_vertex):
        for j in range(num_vertex):
            if i != j:
                ancestor[i][j] = closure[i][j]

    return ancestor
