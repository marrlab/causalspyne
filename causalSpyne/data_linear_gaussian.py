"""
generate Linear Gaussian
"""
import numpy as np
from causalSpyne.noise_idiosyncratic import Gaussian, HyperPars


class DataGenLinearGaussian():
    def __init__(self, dag):
        self.dag = dag

    def gen(self, num_samples, noise_std=1.0):
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
            list_parents_inds = self.dag.get_list_parents_inds(ind_node)
            noise = Gaussian(HyperPars().gen()).gen(num_samples)
            data[:, node] = noise
            if list_parents_inds:
                # Linear combination of parent nodes + Gaussian noise
                weights = self.dag.get_weights_from_list_parents(node)
                bias = np.dot(data[:, list_parents_inds], weights)
                data[:, node] += bias
        return data
