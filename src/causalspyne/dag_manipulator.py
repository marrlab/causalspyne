"""
insert edges to have more confounders
"""
import numpy as np


class DAGManipulator:
    """
    insert edges to have more confounders
    """
    def __init__(self, dag, obj_gen_weight, rng):
        self.dag = dag
        self._obj_gen_weight = obj_gen_weight
        self.rng = rng

    def mk_confound(self, ind_arbitrary_confound_input=None,
                    continue_top_up=True,
                    skip_sink=False):
        """
        ind_arbitrary_confound_input is the arbitrary index for a node, if
        does not set as argument, a random index will be generated, the
        function make the current vertex confounder
        continue_top_up: if not successful, move up toplogically to try again
        skip_sink: if a node is sink node (can be a high toplogical order),
        then does not add edge to other nodes with lower toplogical order
        """
        ind_arbitrary_confound = ind_arbitrary_confound_input
        if ind_arbitrary_confound is None:
            ind_arbitrary_confound = \
                self.rng.choice(self.dag.list_ind_nodes_sorted)
            # list_ind_nodes_sorted looks like [8, 3, 10, 9, 100],
            # each entry is an arbitrary index/name for the node
        # rembmer (i,j) entry means there is arrow from j to i
        # count how many children this current node has
        nnzero = np.count_nonzero(
            self.dag.mat_adjacency[:, ind_arbitrary_confound])
        flag_success = False
        if nnzero == 0:  # 0 means sink node
            # sink node can also be quite high in toplogical rank
            if not skip_sink:
                flag_success_1 = self.add_new_edge(ind_arbitrary_confound)
                flag_success_2 = self.add_new_edge(ind_arbitrary_confound)
                flag_success = flag_success_1 & flag_success_2
        elif nnzero == 1:  # not a sink node but only parent a single child
            # add another child to this current node
            flag_success = self.add_new_edge(ind_arbitrary_confound)
            # not successul in making this node to have two children
        else:  # already a confounder
            flag_success = False  # only when new confounder
        if flag_success:
            return True
        # now flag_success must be False
        if continue_top_up:
            # increase the toplogical order and try until success
            pos = self.dag.global_arbitrary_ind2topind(ind_arbitrary_confound)
            if pos - 1 < 0:
                return False
            ind_arbitrary = self.dag.top_ind2global_arbitrary(pos - 1)
            return self.mk_confound(
                ind_arbitrary_confound_input=ind_arbitrary)
        return False

    def add_new_edge(self, ind_arbitrary_confound):
        """
        insert a new edge pointing to lower toplogical order
        """
        # get toplogical order of this node
        pos = self.dag.global_arbitrary_ind2topind(ind_arbitrary_confound)
        # randomly choose one node which ranked later than the
        # current node
        if pos + 1 == self.dag.num_nodes:
            return False
        ind_toplogical_arrow_head = self.rng.integers(
            pos + 1, self.dag.num_nodes)
        ind_arbitrary_arrow_head = self.dag.top_ind2global_arbitrary(
            ind_toplogical_arrow_head)

        if (
            self.dag.mat_adjacency[
                ind_arbitrary_arrow_head, ind_arbitrary_confound]
            == 0
        ):
            self.dag.mat_adjacency[
                ind_arbitrary_arrow_head, ind_arbitrary_confound
            ] = self._obj_gen_weight.gen(1)
            self.dag.check()  # check if still a DAG
            return True
        # in very rare cases, the randomly chosen node is already a
        # child of this current node
        return False
