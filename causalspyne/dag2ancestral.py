"""
turn DAG into ancestral given list of variabels to hide
"""
import itertools
import copy
import numpy as np
from causalspyne.utils_closure import ancestor_matrix_transpose


def pairwise_combinations(lst):
    """
    generate pairwise combinations from a list
    """
    return itertools.combinations(lst, 2)


class DAG2Ancestral:
    """
    turn DAG into ancestral given list of variabels to hide
    """
    def __init__(self, adj):
        self.old_adj = copy.deepcopy(adj)
        self.mat4ancestral = copy.deepcopy(self.old_adj)
        self.mat_ancestor = None

    def pre_cal_n_hop(self):
        """
        check if one node is ancestor of another
        """
        mat = ancestor_matrix_transpose(self.old_adj)
        self.mat_ancestor = np.transpose(mat)

    def run(self, list_hidden):
        self.pre_cal_n_hop()
        for h in list_hidden:
            self.deal_children(h)
            self.deal_parent(h)
        # delete first axis
        temp_mat_row = np.delete(self.mat4ancestral, list_hidden, axis=0)

        # delete second axis
        mat_adj_subgraph = np.delete(temp_mat_row, list_hidden, axis=1)
        self.mat4ancestral = mat_adj_subgraph
        return self.mat4ancestral

    def is_ancestor(self, c1, c2):
        flag = self.mat_ancestor[c2, c1] > 0
        return flag

    def deal_parent(self, h):
        list_parents = self.get_list_parents(h)
        list_children = self.get_list_children(h)
        for global_ind_parent in list_parents:
            for global_ind_child in list_children:
                self.mat4ancestral[global_ind_child, global_ind_parent] = 1

    def deal_children(self, h):
        """
        for d_1, d_2 in children(h) and d_1, d_2 not connected
        """
        list_children = self.get_list_children(h)
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

    def get_list_children(self, h):
        """
        adj[i,j] indicate arrow from j to i
        """
        column_index = h
        arr = self.old_adj
        nonzero_indices = np.nonzero(arr[:, column_index])[0]
        # np.nonzero() returns a tuple of arrays.
        # Each array in this tuple corresponds to a dimension of
        # the input array and contains the indices of non-zero elements
        # along that dimension.
        # nonzero_elements = arr[nonzero_indices, column_index]
        return list(nonzero_indices)

    def get_list_parents(self, h):
        """
        adj[i,j] indicate arrow from j to i
        """
        column_index = h
        arr = self.old_adj
        nonzero_indices = np.nonzero(arr[column_index, :])[0]
        # np.nonzero() returns a tuple of arrays.
        # Each array in this tuple corresponds to a dimension of
        # the input array and contains the indices of non-zero elements
        # along that dimension.
        # nonzero_elements = arr[nonzero_indices, column_index]
        return list(nonzero_indices)
