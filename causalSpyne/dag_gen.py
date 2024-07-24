"""
concrete class to generate simple DAGs
"""
import numpy as np
from causalSpyne.dag_interface import MatDAG


class Erdos_Renyi():
    """
    uniformly (w.r.t. each edge) decide if the edge will exist w.r.t. a prob.
    threshold
    """
    def __init__(self, prob_neg_weights=0.5):
        """
        """
        self.prob_neg_weights = prob_neg_weights

    def __call__(self, num_nodes, degree, list_weight_range):
        prob = float(degree) / (num_nodes - 1)
        # lower triagular, k=-1 is the lower off diagonal
        mat_lower_triangle_binary = np.tril((
            np.random.rand(num_nodes, num_nodes) < prob).astype(float),
            k=-1)
        # permutes first axis only
        mat_perm = np.random.permutation(np.eye(num_nodes, num_nodes))
        mat_b_permuted = mat_perm.T.dot(
            mat_lower_triangle_binary).dot(mat_perm)
        mat_weight = np.random.uniform(low=list_weight_range[0],
                                       high=list_weight_range[1],
                                       size=[num_nodes, num_nodes])

        # set some edges randomly to negative: e.g. x_i = 2x_j - 3x_k
        mat_weight[np.random.rand(num_nodes, num_nodes)
                   < self.prob_neg_weights] *= -1

        mat_mask = (mat_b_permuted != 0).astype(float)
        # Hardarmard product
        mat_weighted_adjacency = mat_mask * mat_weight
        return mat_weighted_adjacency, mat_mask


class GenDAG():
    def __init__(self, num_nodes, degree, list_weight_range):
        """
        degree: expected degree for each node
        """
        self.num_nodes = num_nodes
        self.degree = degree
        self.list_weight_range = list_weight_range
        self.stategy = Erdos_Renyi()

    def gen_dag(self, num_nodes=None):
        """
        generate DAG and wrap it around with interface
        """
        if num_nodes is None:
            num_nodes = self.num_nodes
        matrix, _ = self.stategy(
            num_nodes, self.degree, self.list_weight_range)
        dag = MatDAG(matrix)
        return dag
