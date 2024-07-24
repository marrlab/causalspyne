"""
class method for DAG operations and check
"""
import random


class MatDAG():
    """
    DAG represented as a mat_adjacency
    """
    def __init__(self, mat_adjacency):
        """
        """
        self.mat_adjacency = mat_adjacency

    @property
    def arcs(self):
        """
        return the list of edges
        """
        return list(zip(*self.mat_adjacency.nonzero()))

    def sample_node(self, ind_macro):
        """
        randomly chose a node
        """
        num = random.randint(0, self.mat_adjacency.shape[0] - 1)
        return num * (ind_macro + 1)

    def add_arc(self, ind_tail, ind_head):
        """
        add edges
        """
