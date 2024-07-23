class MatDAG():
    def __init__(self, matrix):
        """
        """
        self.matrix = matrix

    @property
    def arcs(self):
        """
        """
        return list(zip(*self.matrix.nonzero()))

    def sample_node(self, ind_macro):
        """
        """
        random.randint(self.matrx.shape[0]) * (ind_macro + 1)

    def add_arc(self, ind_tail, ind_head):
        """
        """
