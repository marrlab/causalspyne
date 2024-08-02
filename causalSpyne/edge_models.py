import numpy as np


class EdgeModelLinear():
    def __init__(self, dag):
        self._dag = dag

    def run(self, ind_node, data_realization_parents):
        """
        data_realization_parents = data[:, list_parents_inds]
        """
        # Linear combination of parent nodes + Gaussian noise
        weights = self._dag.get_weights_from_list_parents(ind_node)
        bias = np.dot(data_realization_parents, weights)
        return bias
