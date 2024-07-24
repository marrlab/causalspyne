"""
class method for DAG operations and check
"""
import random
from scipy.linalg import block_diag


class MatDAG():
    """
    DAG represented as a mat_adjacency
    """
    def __init__(self, mat_adjacency, name_prefix=""):
        """
        """
        self.name_prefix = name_prefix
        self.mat_adjacency = mat_adjacency
        self._list_node_names = None
        self._dict_node_names2ind = {}

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
        self._list_node_names = [str(i) for i in range(self.num_nodes)]
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

    @classmethod
    def stack_dags(cls, dict_dags):
        """j
        stack dictionary of DAG into a block diagnoal matrix
        """
        mat_stacked_dag = block_diag(
            *(dag.mat_adjacency for dag in dict_dags.values()))
        return MatDAG(mat_stacked_dag)

    def __str__(self):
        return str(self.mat_adjacency)

    def __repr__(self):
        return str(self.mat_adjacency)
