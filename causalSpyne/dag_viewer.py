"""
create different views for the same DAG by hiding some variables
"""


class DAGView():
    def __init__(self, dag):
        self._dag = dag

    def hide(self, toplogical_order_ind_node):
        global_ind = self._dag.list_ind_nodes_sorted[toplogical_order_ind_node]