import numpy as np


class WeightGenUniform():
    def __init__(self, list_weight_range, prob_neg_weights=0.5):
        self.list_weight_range = list_weight_range
        self.prob_neg_weights = prob_neg_weights

    def gen(self, num_nodes):
        """
        generate complete graph, fully connected
        """
        mat_weight = np.random.uniform(low=self.list_weight_range[0],
                                       high=self.list_weight_range[1],
                                       size=[num_nodes, num_nodes])

        # set some edges randomly to negative: e.g. x_i = 2x_j - 3x_k
        mat_weight[np.random.rand(num_nodes, num_nodes)
                   < self.prob_neg_weights] *= -1
        if mat_weight.size == 1:
            return mat_weight.item()
        return mat_weight
