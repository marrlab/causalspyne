"""
class method for DAG operations and check
"""
import random
import numpy as np
from causalSpyne.is_dag import is_dag
from causalSpyne.utils_topological_sort import topological_sort


def add_prefix(string, prefix="", separator="_"):
    """
    Adds a prefix to a string.
    If the prefix is empty, returns the original string.

    Args:
    string (str): The original string.
    prefix (str, optional): The prefix to add. Defaults to an empty string.

    Returns:
    str: The string with the prefix added.
    """
    if not prefix:
        return string
    return separator.join([prefix, string])


class MatDAG():
    """
    DAG represented as a mat_adjacency
    """
    def __init__(self, mat_adjacency, name_prefix="", separator="_"):
        """
        """
        self.separator = separator
        self.name_prefix = name_prefix
        self.mat_adjacency = mat_adjacency
        self._list_node_names = None
        self._dict_node_names2ind = {}
        self.list_ind_nodes_sorted = None

    def check(self):
        """
        check if the matrix represent a DAG
        """
        if not is_dag(self.mat_adjacency):
            raise RuntimeError("not a DAG")

    @property
    def num_nodes(self):
        """
        number of nodes in DAG
        """
        return self.mat_adjacency.shape[0]

    def gen_node_names(self):
        """
        get list of node names
        """
        self._list_node_names = [
            add_prefix(string=str(i), prefix=self.name_prefix)
            for i in range(self.num_nodes)]
        self._dict_node_names2ind = \
            {name: i for (i, name) in enumerate(self._list_node_names)}

    def gen_node_names_stacked(self, dict_macro_node2dag):
        self._list_node_names = []
        for key, dag in dict_macro_node2dag.items():
            self._list_node_names.extend(dag.list_node_names)
        self._dict_node_names2ind = \
            {name: i for (i, name) in enumerate(self._list_node_names)}

    def get_node_ind(self, node_name):
        """
        get the matrix index of the node name
        """
        return self._dict_node_names2ind[node_name]

    @property
    def list_node_names(self):
        """
        get the node names in a linear list
        """
        if self._list_node_names is None:
            self.gen_node_names()
        return self._list_node_names

    @property
    def list_arcs(self):
        """
        return the list of edges
        """
        list_i_j = list(zip(*self.mat_adjacency.nonzero()))
        list_arcs = \
            [(self.list_node_names[tuple(ij)[0]],
              self.list_node_names[tuple(ij)[1]]) for ij in list_i_j]
        return list_arcs

    def sample_node(self):
        """
        randomly chose a node
        """
        ind = random.randint(0, self.mat_adjacency.shape[0] - 1)
        name = self.list_node_names[ind]
        return self.name_prefix + name, ind

    def add_arc_ind(self, ind_tail, ind_head, weight=None):
        """
        add arc via index of tail and head
        """
        node_tail = self.list_node_names[ind_tail]
        node_head = self.list_node_names[ind_head]
        self.add_arc(node_tail, node_head, weight)

    def add_arc(self, node_tail, node_head, weight=None):
        """
        add edge to adjacency matrix
        """
        ind_tail = self._dict_node_names2ind[node_tail]
        ind_head = self._dict_node_names2ind[node_head]
        if weight is None:
            self.mat_adjacency[ind_tail, ind_head] = 1
        else:
            self.mat_adjacency[ind_tail, ind_head] = weight

    def topological_sort(self):
        """
        topological sort DAG into list of node index
        """
        binary_adj_mat = (self.mat_adjacency != 0).astype(int)
        self.list_ind_nodes_sorted = topological_sort(binary_adj_mat)
        return self.list_ind_nodes_sorted

    def get_list_parents_inds(self, ind_node):
        """
        get list of parents nodes
        """
        # np.nonzero(x)
        # returns (array([0, 1, 2, 2]), array([0, 1, 0, 1]))
        # assume lower triangular matrix as adjacency matrix
        # matrix[i, j]=1 indicate arrow j->i
        submatrix = self.mat_adjacency[ind_node, :]
        vector = submatrix.flatten()
        list_inds = np.nonzero(vector)[0].tolist()
        return list_inds

    def get_weights_from_list_parents(self, ind_sink, list_parents):
        # FIXME: list_parents are redundant, since we could get the parent
        # of ind_insk
        """
        get incoming edge weights
        """
        sub_matrix = self.mat_adjacency[ind_sink, list_parents]
        return sub_matrix

    def __str__(self):
        return str(self.mat_adjacency)

    def __repr__(self):
        return str(self.mat_adjacency)
