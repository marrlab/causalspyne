"""
concrete class to generate simple DAGs
"""
import numpy as np


class Erdos_Renyi():
    """
    uniformly (w.r.t. each edge) decide if the edge will exist w.r.t. a prob.
    threshold
    """
    def __call__(self, num_nodes, degree, list_weight_range):
        prob = float(degree) / (num_nodes - 1)
        # lower triagular, k=-1 is the lower off diagonal
        mat_lower_triangle = np.tril((
            np.random.rand(num_nodes, num_nodes) < prob).astype(float),
            k=-1)
        # permutes first axis only
        mat_perm = np.random.permutation(np.eye(num_nodes, num_nodes))
        mat_b_permuted = mat_perm.T.dot(mat_lower_triangle).dot(mat_perm)
        mat_weight = np.random.uniform(low=list_weight_range[0], high=list_weight_range[1],
                                       size=[num_nodes, num_nodes])

        mat_weight[np.random.rand(num_nodes, num_nodes) < 0.5] *= -1

        mat_mask = (mat_b_permuted != 0).astype(float)
        # Hardarmard product
        mat_weighted_adjacency =  mat_mask * mat_weight
        return mat_weighted_adjacency, mat_mask


class GenDAGER():
    def __init__(self, num_nodes, degree, list_weight_range):
        self.num_nodes = num_nodes
        self.degree = degree
        self.list_weight_range = list_weight_range
        self.gen = Erdos_Renyi()

    def genDAG(self, num_nodes=None):
        if num_nodes is None:
            num_nodes = self.num_nodes
        matrix, _ = self.gen(num_nodes, self.degree, self.list_weight_range)
        return matrix


def test_erdos_renyi():
    Erdos_Renyi()(num_nodes=3, degree=2, list_weight_range=[3, 5])
