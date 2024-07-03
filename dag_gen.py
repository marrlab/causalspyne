import numpy as np


class Erdos_Renyi():
    def __call__(self, num_nodes, degree, weight_range):
        prob = float(degree) / (num_nodes - 1)
        # lower triagular, k=-1 is the lower off diagonal
        mat_b = np.tril((
            np.random.rand(num_nodes, num_nodes) < prob).astype(float),
            k=-1)
        mat_perm = np.random.permutation(np.eye(num_nodes, num_nodes))  # permutes first axis only
        mat_b_permuted = mat_perm.T.dot(mat_b).dot(mat_perm)
        mat_weight = np.random.uniform(low=weight_range[0], high=weight_range[1],
                                       size=[num_nodes, num_nodes])
        mat_weight[np.random.rand(num_nodes, num_nodes) < 0.5] *= -1

        W = (mat_b_permuted != 0).astype(float) * mat_weight
        return W, mat_weight


class GenDAGER():
    def __init__(self, num_nodes, degree, weight_range):
        self.num_nodes = num_nodes
        self.degree = degree
        self.weight_range = weight_range
        self.gen = Erdos_Renyi()

    def genDAG(self, num_nodes=None):
        if num_nodes is None:
            num_nodes = self.num_nodes
        matrix, _ = self.gen(num_nodes, self.degree, self.weight_range)
        return matrix


def test_erdos_renyi():
    Erdos_Renyi()(num_nodes=3, degree=2, weight_range=[3, 5])
