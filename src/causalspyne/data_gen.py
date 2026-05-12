"""
generate Linear Gaussian
"""

import numpy as np

from causalspyne.noise_idiosyncratic import Idiosyncratic
from causalspyne.edge_models import EdgeModelLinear
from causalspyne.utils_random import coerce_rng


class DataGen:
    def __init__(self, dag, edge_model=None,
                 dft_noise: str="Gaussian",
                 dict_params: dict | None=None,
                 idiosynchratic: dict[int, Idiosyncratic] | None=None,
                 rng=None):
        rng = coerce_rng(rng, seed=0)
        self.dag = dag
        self.dict_idiosyncratic = idiosynchratic or {}

        self.idiosyncratic = Idiosyncratic(class_name=dft_noise,
                                           rng=rng,
                                           dict_params=dict_params or {})
        self.edge_model = edge_model
        if edge_model is None:
            self.edge_model = EdgeModelLinear(self.dag)

    def gen(self, num_samples):
        """
        Generate linear Gaussian data from a given DAG.

        Parameters:
        - num_samples: int, number of samples to generate.
        - noise_std: float, standard deviation of the Gaussian noise.

        Returns:
        - data: np.ndarray, generated data of shape (num_samples, num_nodes).
        """
        list_ind_nodes_topo_order = self.dag.topological_sort()

        # Number of nodes
        num_nodes = len(list_ind_nodes_topo_order)

        # Initialize the data matrix
        data = np.zeros((num_samples, num_nodes))

        # Generate data for each node in topological order
        for node in list_ind_nodes_topo_order:
            list_parents_inds = self.dag.get_list_parents_inds(node)
            if node in self.dict_idiosyncratic:
                noise = self.dict_idiosyncratic[node].gen(num_samples)
            else:
                noise = self.idiosyncratic.gen(num_samples)
            data[:, node] = noise
            if list_parents_inds:
                bias = self.edge_model.run(node, data[:, list_parents_inds])
                data[:, node] += bias
        return data
