"""
concrete class to generate simple DAGs
"""
import numpy as np
from causalSpyne.dag_interface import MatDAG
from causalSpyne.weight import WeightGenUniform


class Erdos_Renyi_PLP():
    """
    uniformly (w.r.t. each edge) decide if the edge will exist w.r.t. a prob.
    threshold
    trick: PLP^T to ensure a DAG, L is lower triangular (topological order)
    P is permutation, P permute the labels
    (i,j) entry indicate j->i
    row permutation  (i,j), (k,j) becomes (k,j) (i,j)
    column permutation
    """
    def __call__(self, num_nodes, degree):
        prob = float(degree) / (num_nodes - 1)
        # lower triagular, k=-1 is the lower off diagonal
        mat_lower_triangle_binary = np.tril((
            np.random.rand(num_nodes, num_nodes) < prob).astype(float),
            k=-1)
        # permutes first axis only
        mat_perm = np.random.permutation(np.eye(num_nodes, num_nodes))
        mat_b_permuted = mat_perm.T.dot(
            mat_lower_triangle_binary).dot(mat_perm)
        return mat_b_permuted


class GenDAG():
    def __init__(self, num_nodes, degree, list_weight_range):
        """
        degree: expected degree for each node
        """
        self.num_nodes = num_nodes
        self.degree = degree
        self.list_weight_range = list_weight_range
        self.stategy_gen_dag = Erdos_Renyi_PLP()
        self.obj_gen_weight = WeightGenUniform(list_weight_range)

    def gen_dag(self, num_nodes=None, prefix=""):
        """
        generate DAG and wrap it around with interface
        """
        if num_nodes is None:
            num_nodes = self.num_nodes
        mat_skeleton = self.stategy_gen_dag(
            num_nodes, self.degree)

        mat_mask = (mat_skeleton != 0).astype(float)
        mat_weight = self.obj_gen_weight.gen(num_nodes)
        # Hardarmard product
        mat_weighted_adjacency = mat_mask * mat_weight

        dag = MatDAG(mat_weighted_adjacency, name_prefix=prefix)
        return dag
