import numpy as np

class DAGManipulator():
    def __init__(self, dag, obj_gen_weight):
        self.dag = dag
        self._obj_gen_weight = obj_gen_weight

    def mk_confound(self, ind):
        """
        make the current vertex confounder
        """
        nnzero = np.count_nonzero(self.dag.mat_adjacency[:, ind])
        if nnzero < 2:
            pos = self.list_ind_nodes_sorted.index(ind)
            j = random.randint(pos + 1, self.num_nodes - 1)
            self.mat_adjacency[j, ind] = self._obj_gen_weight()


