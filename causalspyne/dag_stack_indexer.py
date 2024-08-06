"""
index node globally when stacking several dags together
"""
from scipy.linalg import block_diag
from causalspyne.dag_interface import MatDAG


class DAGStackIndexer():
    """
    stack DAG indexer
    """
    def __init__(self, host):
        """
        """
        self.host = host
        self.host.dag_refined = self.stack_dags()

    def get_global_ind(self, ind_macro_node, ind_local_node):
        """
        get the global index of a local node
        """
        return self.list_accum_count[ind_macro_node] + ind_local_node

    def stack_dags(self):
        """
        stack dictionary of DAG into a block diagnoal matrix
        """
        mat_stacked_dag = block_diag(
            *(dag.mat_adjacency for dag in
              self.host.dict_macro_node2dag.values()))
        dag_stacked = MatDAG(mat_stacked_dag)
        self.dict_num = {key: dag.num_nodes
                         for key, dag in self.host.dict_macro_node2dag.items()}
        list_accum_count = list(fun_accum_sum(self.dict_num))
        self.list_accum_count = [0] + list_accum_count[:-1]
        dag_stacked.gen_node_names_stacked(self.host.dict_macro_node2dag)
        return dag_stacked


def fun_accum_sum(dict_num):
    """
    count accumulative sum of a dictionary
    e.g. dict_dag = {'0':3, '1':5, '2':4}
    """
    count = 0
    for num in dict_num.values():
        count += num
        yield count
