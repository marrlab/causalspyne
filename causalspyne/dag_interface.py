"""
class method for DAG operations and check
"""
import random
import numpy as np
import pandas as pd
from causalspyne.is_dag import is_dag
from causalspyne.utils_topological_sort import topological_sort
from causalspyne.draw_dags import draw_dags_nx


def add_prefix(string, prefix="", separator="u"):
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
    def __init__(self, mat_adjacency, name_prefix="",
                 separator="_", list_node_names=None):
        """
        """
        self._obj_gen_weight = None
        self.separator = separator
        self.name_prefix = name_prefix
        self.mat_adjacency = mat_adjacency
        self._list_node_names = list_node_names
        self._list_confounder = None
        self._dict_node_names2ind = {}
        self._init_map()
        self._list_ind_nodes_sorted = None

    def _init_map(self):
        if self._list_node_names is not None:
            self._dict_node_names2ind = \
                {name: i for (i, name) in enumerate(self._list_node_names)}

    @property
    def list_confounder(self):
        """
        return list of confounders
        """
        nonzero_counts = np.count_nonzero(self.mat_adjacency, axis=0)
        columns_with_more_than_one = np.where(nonzero_counts > 1)[0]
        return list(columns_with_more_than_one)

    def gen_dict_ind2node_na(self):
        """
        utility function to have {1:"node_name"} dictionary for plotting
        """
        mdict = {i: name for (i, name) in enumerate(self.list_node_names)}
        return mdict

    def check(self):
        """
        check if the matrix represent a DAG
        """
        if not is_dag(self.mat_adjacency):
            raise RuntimeError("not a DAG")
        binary_adj_mat = (self.mat_adjacency != 0).astype(int)
        if not is_dag(binary_adj_mat):
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
            add_prefix(string="v" + str(i), prefix=self.name_prefix)
            for i in range(self.num_nodes)]
        self._init_map()

    def gen_node_names_stacked(self, dict_macro_node2dag):
        self._list_node_names = []
        for key, dag in dict_macro_node2dag.items():
            self._list_node_names.extend(dag.list_node_names)
        self._init_map()

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

    def to_binary_csv(self, benchpress=True):
        """
        adjacency matrix to csv format
        """
        binary_adj_mat = (self.mat_adjacency != 0).astype(int)
        if benchpress:
            binary_adj_mat = np.transpose(binary_adj_mat)
        df = pd.DataFrame(binary_adj_mat, columns=self.list_node_names)
        df.to_csv("adj.csv", index=False)

    def topological_sort(self):
        """
        topological sort DAG into list of node index
        """
        binary_adj_mat = (self.mat_adjacency != 0).astype(int)
        self._list_ind_nodes_sorted = topological_sort(binary_adj_mat)
        return self._list_ind_nodes_sorted

    @property
    def list_ind_nodes_sorted(self):
        """
        get global node index topologically sorted
        """
        if self._list_ind_nodes_sorted is None:
            self.topological_sort()
        return self._list_ind_nodes_sorted

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

    def get_weights_from_list_parents(self, ind_node):
        """
        get incoming edge weights
        """
        list_parents_inds = self.get_list_parents_inds(ind_node)
        sub_matrix = self.mat_adjacency[ind_node, list_parents_inds]
        return sub_matrix

    def __str__(self):
        return str(self.mat_adjacency)

    def __repr__(self):
        return str(self.mat_adjacency)

    def subgraph(self, list_ind_unobserved):
        """
        subset adjacency matrix by deleting unobserved variables
        """
        temp_mat_row = np.delete(
            self.mat_adjacency, list_ind_unobserved, axis=0)
        mat_adj_subgraph = np.delete(
            temp_mat_row, list_ind_unobserved, axis=1)
        list_node_names_subgraph = [x for i, x in
                                    enumerate(self.list_node_names)
                                    if i not in list_ind_unobserved]
        subdag = MatDAG(mat_adj_subgraph,
                        list_node_names=list_node_names_subgraph)
        return subdag

    def visualize(self, title="dag"):
        """
        draw dag using networkx
        """
        draw_dags_nx(self.mat_adjacency,
                     dict_ind2name=self.gen_dict_ind2node_na(),
                     title=title)

    @property
    def list_top_names(self):
        """
        return list of node names in toplogical order
        """
        list_top_names = [self.list_node_names[i]
                          for i in self.list_ind_nodes_sorted]
        return list_top_names
