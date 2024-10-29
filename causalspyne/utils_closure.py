import numpy as np


def transitive_closure(adj_matrix):
    """
    Floyd-Warshall algorithm, note that this algorithm is agnostic w.r.t.
    the edge convention: (i,j) implies either i<-j or i->j is the same
    for the algorithm
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
                # here the mathematical relation i-j or i~j, can mean
                # both i->j or j<-i
    return closure


def ancestor_matrix(adj_matrix):
    num_vertex = len(adj_matrix)
    closure = transitive_closure(adj_matrix)
    ancestor = np.zeros((num_vertex, num_vertex), dtype=bool)

    for i in range(num_vertex):
        for j in range(num_vertex):
            if i != j:
                ancestor[i][j] = closure[i][j]
    return ancestor
