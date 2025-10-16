from numpy.random import default_rng

from causalspyne.wishart import gen_weight_matrix


class WeightGenUniform:
    def __init__(self, list_weight_range, prob_neg_weights=0.5, rng=default_rng(0)):
        self.list_weight_range = list_weight_range
        self.prob_neg_weights = prob_neg_weights
        self.rng = rng

    def gen(self, num_nodes):
        """
        generate complete graph, fully connected
        """
        mat_weight = self.rng.uniform(
            low=self.list_weight_range[0],
            high=self.list_weight_range[1],
            size=[num_nodes, num_nodes],
        )

        # set some edges randomly to negative: e.g. x_i = 2x_j - 3x_k
        random_mask = self.rng.choice(
            [True, False],
            (num_nodes, num_nodes),
            (self.prob_neg_weights, 1 - self.prob_neg_weights),
        )
        mat_weight[random_mask] *= -1
        if mat_weight.size == 1:
            return mat_weight.item()
        return mat_weight


class WeightGenWishart(WeightGenUniform):
    def __init__(self, prob_neg_weights=0.5, rng=default_rng(0)):
        self.prob_neg_weights = prob_neg_weights
        self.rng = rng

    def gen(self, num_nodes):
        """
        generate complete graph, fully connected
        """
        mat_weight = gen_weight_matrix(self.rng, num_nodes)

        # set some edges randomly to negative: e.g. x_i = 2x_j - 3x_k
        random_mask = self.rng.choice(
            [True, False],
            (num_nodes, num_nodes),
            (self.prob_neg_weights, 1 - self.prob_neg_weights),
        )
        mat_weight[random_mask] *= -1
        if mat_weight.size == 1:
            return mat_weight.item()
        return mat_weight
