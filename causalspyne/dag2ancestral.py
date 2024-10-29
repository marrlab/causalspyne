"""
turn DAG into ancestral given list of variabels to hide
"""

import itertools
import copy
import numpy as np
from causalspyne.utils_closure import ancestor_matrix


def pairwise_combinations(lst):
    """
    generate pairwise combinations from a list
    """
    return itertools.combinations(lst, 2)


def to_binary(matrix):
    """
    convert a matrix to 0,1 matrix
    """
    binary_matrix = (matrix != 0).astype(int)
    return binary_matrix


class DAG2Ancestral:
    """
    turn DAG into ancestral given list of variabels to hide
    """

    def __init__(self, adj):
        self.old_adj = copy.deepcopy(adj)
        self.mat4ancestral = copy.deepcopy(self.old_adj)
        self.bmat_ancestor = None

    def pre_cal_n_hop(self):
        """
        check if one node is ancestor of another
        """
        self.bmat_ancestor = ancestor_matrix(self.old_adj)

    def run(self, list_hidden):
        """
        convert DAG to ancestral
        """
        self.pre_cal_n_hop()
        for hidden in list_hidden:
            self.deal_children(hidden)
            self.deal_parent(hidden)
        # delete first axis
        temp_mat_row = np.delete(self.mat4ancestral, list_hidden, axis=0)
        # delete second axis
        mat_adj_subgraph = np.delete(temp_mat_row, list_hidden, axis=1)
        self.mat4ancestral = mat_adj_subgraph
        return to_binary(self.mat4ancestral)

    def is_ancestor(self, global_ind_node_1, global_ind_node_2):
        """
        check if the first argument is an ancestor of the second
        """
        flag = self.bmat_ancestor[global_ind_node_2, global_ind_node_1]
        return flag

    def deal_parent(self, hidden):
        """
        connect parent of hidden and child of hidden
        """
        list_parents = self.get_list_parents(hidden)
        list_children = self.get_list_children(hidden)
        for global_ind_parent in list_parents:
            for global_ind_child in list_children:
                self.mat4ancestral[global_ind_child, global_ind_parent] = 1

    def deal_children(self, hidden):
        """
        for d_1, d_2 in children(hidden) and d_1, d_2 not connected
        """
        list_children = self.get_list_children(hidden)
        if len(list_children) < 2:
            return
        for pair in pairwise_combinations(list_children):
            c1_global_ind, c2_global_ind = pair
            if self.is_ancestor(c1_global_ind, c2_global_ind):
                # mat[i,j] means edge from j to i
                self.mat4ancestral[c2_global_ind, c1_global_ind] = 1
            elif self.is_ancestor(c2_global_ind, c1_global_ind):
                self.mat4ancestral[c1_global_ind, c2_global_ind] = 1
            else:
                self.mat4ancestral[c1_global_ind, c2_global_ind] = 1
                self.mat4ancestral[c2_global_ind, c1_global_ind] = 1

    def get_list_children(self, hidden):
        """
        adj[i,j] indicate arrow from j to i
        """
        arr = self.old_adj
        nonzero_indices = np.flatnonzero(arr[:, hidden])
        # np.nonzero() returns a tuple of arrays.
        # Each array in this tuple corresponds to a dimension of
        # the input array and contains the indices of non-zero elements
        # along that dimension.
        # nonzero_elements = arr[nonzero_indices, column_index]
        list_non_zero_indices = nonzero_indices.tolist()
        return list_non_zero_indices

    def get_list_parents(self, hidden):
        """
        adj[i,j] indicate arrow from j to i
        """
        arr = self.old_adj
        nonzero_indices = np.flatnonzero(arr[hidden, :])
        # np.nonzero() returns a tuple of arrays.
        # Each array in this tuple corresponds to a dimension of
        # the input array and contains the indices of non-zero elements
        # along that dimension.
        # nonzero_elements = arr[nonzero_indices, column_index]
        list_non_zero_indices = nonzero_indices.tolist()
        return list_non_zero_indices
