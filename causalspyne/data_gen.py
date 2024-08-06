"""
generate Linear Gaussian
"""
import numpy as np
from causalspyne.noise_idiosyncratic import Idiosyncratic
from causalspyne.edge_models import EdgeModelLinear


class DataGen():
    def __init__(self, dag, edge_model=None, idiosynchratic=None):
        self.dag = dag
        self.idiosyncratic = idiosynchratic
        if idiosynchratic is None:
            self.idiosyncratic = Idiosyncratic()
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
            noise = self.idiosyncratic.gen(num_samples)
            data[:, node] = noise
            if list_parents_inds:
                bias = self.edge_model.run(node, data[:, list_parents_inds])
                data[:, node] += bias
        return data
