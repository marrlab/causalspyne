import random
import numpy as np


class DAGManipulator():
    def __init__(self, dag, obj_gen_weight):
        self.dag = dag
        self._obj_gen_weight = obj_gen_weight

    def mk_confound(self, ind_arbitrary_confound=None):
        """
        make the current vertex confounder
        """
        if ind_arbitrary_confound is None:
            ind_arbitrary_confound = random.choices(self.dag.list_ind_nodes_sorted)[0]
        nnzero = np.count_nonzero(self.dag.mat_adjacency[:, ind_arbitrary_confound])
        if nnzero == 0:  # 0 means sink node
            return False
        if nnzero == 1:
            pos = self.dag.list_ind_nodes_sorted.index(ind_arbitrary_confound)
            ind_toplogical_arrow_head = random.randint(pos + 1, self.dag.num_nodes - 1)
            ind_arbitrary_arrow_head = self.dag.list_ind_nodes_sorted[ind_toplogical_arrow_head]
            # pos + 1 ensures ind_arbitrary_confound != ind_arbitrary_arrow_head
            if self.dag.mat_adjacency[ind_arbitrary_arrow_head, ind_arbitrary_confound] == 0:
                self.dag.mat_adjacency[ind_arbitrary_arrow_head, ind_arbitrary_confound] = \
                    self._obj_gen_weight.gen(1)
                self.dag.check()
            else:
                return False
        return True
