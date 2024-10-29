"""
insert edges to have more confounders
"""
import numpy as np


class DAGManipulator:
    def __init__(self, dag, obj_gen_weight, rng):
        self.dag = dag
        self._obj_gen_weight = obj_gen_weight
        self.rng = rng

    def mk_confound(self, ind_arbitrary_confound_input=None, force=True):
        """
        make the current vertex confounder
        ind_arbitrary_confound_input is the arbitrary index for a node
        """
        ind_arbitrary_confound = ind_arbitrary_confound_input
        if ind_arbitrary_confound is None:
            ind_arbitrary_confound = self.rng.choice(self.dag.list_ind_nodes_sorted)
            # list_ind_nodes_sorted looks like [8, 3, 10, 9, 100], each entry is an
            # arbitrary index/name for the node
        # rembmer (i,j) entry means there is arrow from j to i
        # count how many children this current node has
        nnzero = np.count_nonzero(self.dag.mat_adjacency[:, ind_arbitrary_confound])
        if nnzero == 0:  # 0 means sink node
            # sink node can also be quite high in toplogical rank
            return False
        if nnzero == 1:  # not a sink node but only parent a single child
            # add another child to this current node
            # get toplogical order of this node
            pos = self.dag.list_ind_nodes_sorted.index(ind_arbitrary_confound)
            # randomly choose one node which ranked later than the
            # current node
            ind_toplogical_arrow_head = self.rng.integers(pos + 1, self.dag.num_nodes)
            ind_arbitrary_arrow_head = self.dag.list_ind_nodes_sorted[
                ind_toplogical_arrow_head
            ]
            # pos + 1 ensures ind_arbitrary_confound !=
            # ind_arbitrary_arrow_head
            if (
                self.dag.mat_adjacency[ind_arbitrary_arrow_head,
                                       ind_arbitrary_confound]
                == 0
            ):
                self.dag.mat_adjacency[
                    ind_arbitrary_arrow_head, ind_arbitrary_confound
                ] = self._obj_gen_weight.gen(1)
                self.dag.check()  # check if still a DAG
            else:
                # in very rare cases, the randomly chosen node is already a
                # child of this current node
                if force:
                    # increase the toplogical order and try until success
                    if ind_arbitrary_confound - 1 < 0:
                        return False
                    return self.mk_confound(
                        ind_arbitrary_confound_input=ind_arbitrary_confound-1)
                return False
                # not successul in making this node to have two children
        return True
