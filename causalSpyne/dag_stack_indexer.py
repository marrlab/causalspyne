"""
index node globally when stacking several dags together
"""
from causalSpyne.dag_interface import MatDAG


class DAGStackIndexer():
    """
    stack DAG indexer
    """
    def __init__(self, dict_dag):
        """
        """
        self._dag_refined = MatDAG.stack_dags(dict_dag)
        self.dict_num = {key: dag.num_nodes for key, dag in dict_dag.items()}
        list_accum_count = list(fun_accum_sum(self.dict_num))
        self.list_accum_count = [0] + list_accum_count[:-1]

    @property
    def dag_refined(self):
        """
        return the stacked dag
        """
        return self._dag_refined

    def get_global_ind(self, ind_macro_node, ind_local_node):
        """
        get the global index of a local node
        """
        return self.list_accum_count[ind_macro_node] + ind_local_node


def fun_accum_sum(dict_num):
    """
    count accumulative sum of a dictionary
    e.g. dict_dag = {'0':3, '1':5, '2':4}
    """
    count = 0
    for num in dict_num.values():
        count += num
        yield count
