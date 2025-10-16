"""
concrete class to generate simple DAGs

"""
import numpy as np


class Erdos_Renyi_PLP:
    """
    uniformly (w.r.t. each edge) decide if the edge will exist w.r.t. a prob.
    threshold
    trick: PLP^T to ensure a DAG, L is lower triangular (topological order)
    P is permutation, P permute the labels
    (i,j) entry indicate j->i
    row permutation  (i,j), (k,j) becomes (k,j) (i,j)
    column permutation
    """

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, num_nodes, degree):
        prob = float(degree) / (num_nodes - 1)
        # lower triagular, k=-1 is the lower off diagonal
        mat_lower_triangle_binary = np.tril(
            (self.rng.random((num_nodes, num_nodes)) < prob
             ).astype(float), k=-1
        )
        # permutes first axis only
        mat_perm = self.rng.permutation(np.eye(num_nodes, num_nodes))
        mat_b_permuted = mat_perm.T.dot(
            mat_lower_triangle_binary).dot(mat_perm)
        return mat_b_permuted
