"""
create different views for the same DAG by hiding some variables
"""
import numpy as np
import pandas as pd
from causalSpyne.data_gen import DataGen
from causalSpyne.dag_interface import MatDAG


class DAGView():
    """
    with ground truth DAG intact, only show subgraph
    """
    def __init__(self, dag):
        self._dag = dag
        # there is no need to use a full DAG to represent subdag
        # since sub-dag is not responsible for data generation
        self._sub_dag = None
        self._data_arr = None
        self._subset_data_arr = None
        self._list_global_inds_unobserved = None
        self.data_gen = DataGen(self._dag)

    def run(self, num_samples, list_nodes2hide=None):
        """
        generate subgraph adjcency matrix and corresponding data
        """
        self._data_arr = self.data_gen.gen(num_samples)
        if list_nodes2hide is None:
            list_nodes2hide = [0]
        self.hide(list_nodes2hide)

    def hide(self, list_toporder_unobserved):
        """
        hide variables according to a list of global index of topological sort
        """
        # subset list
        nodes2remove = [self._dag.list_top_names[i] for i in list_toporder_unobserved]
        print("nodes to hide " + str(nodes2remove))
        self._list_global_inds_unobserved = [self._dag.list_ind_nodes_sorted[ind_top_order]
             for ind_top_order in list_toporder_unobserved]
        self._sub_dag = self._dag.subgraph(self._list_global_inds_unobserved)
        self._subset_data_arr = np.delete(self._data_arr,
                                          self._list_global_inds_unobserved, axis=1)

    @property
    def data(self):
        return self._subset_data_arr

    @property
    def mat_adj(self):
        return self._sub_dag.mat_adjacency

    def to_csv(self):
        """
        sub dataframe to  csv
        """
        node_names = [name for (i, name) in enumerate(self._dag.list_node_names)
                      if i not in self._list_global_inds_unobserved]
        df = pd.DataFrame(self.data, columns=node_names)
        df.to_csv("output.csv", index=False)
        subdag = MatDAG(self.mat_adj)
        subdag.to_binary_csv()

    def visualize(self, title):
        self._sub_dag.visualize(title=title)
