"""
create different views for the same DAG by hiding some variables
"""
import numpy as np
import pandas as pd
from causalSpyne.data_gen import DataGen
from causalSpyne.dag_interface import MatDAG


def gen_list2hide(list_or_percentage, total_num):
    if isinstance(list_or_percentage, float):
        pos = int(list_or_percentage * total_num)
        list_chosen = list(range(pos, total_num + 1))
        return list_chosen
    return list_or_percentage


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

    def run(self, num_samples, list_nodes2hide=None, confound=False):
        """
        generate subgraph adjcency matrix and corresponding data
        """
        self._data_arr = self.data_gen.gen(num_samples)
        if list_nodes2hide is None:
            list_nodes2hide = [0]
        if confound:
            self.hide_confounder(list_nodes2hide)
        else:
            self.hide(list_nodes2hide)

    def hide_confounder(self, list_toporder_confounder):
        """
        given a list of index, hide the confounder according to the toplogical
        order provided by the input index list_toporder_confounder
        """
        list_toporder_confounder = gen_list2hide(
            list_toporder_confounder, len(self._dag.list_confounder))
        list_toporder_unobserved = \
            [self._dag.list_ind_nodes_sorted.index(confounder)
             for confounder in self._dag.list_confounder]
        if len(list_toporder_confounder) > len(list_toporder_unobserved) or \
                max(list_toporder_confounder) > len(list_toporder_unobserved):
            raise RuntimeError("there are less confounders than the length \
                               of input list_toporder_confounder")
        list_toporder_confounder_sub = \
            [list_toporder_unobserved[i] for i in list_toporder_confounder]
        self.hide(list_toporder_confounder_sub)

    def hide(self, list_toporder_unobserved):
        """
        hide variables according to a list of global index of topological sort
        """
        list_toporder_unobserved = gen_list2hide(
            list_toporder_unobserved, self._dag.num_nodes)

        # subset list
        nodes2remove = [self._dag.list_top_names[i]
                        for i in list_toporder_unobserved]
        print("nodes to hide " + str(nodes2remove))
        self._list_global_inds_unobserved = \
            [self._dag.list_ind_nodes_sorted[ind_top_order]
             for ind_top_order in list_toporder_unobserved]
        self._sub_dag = self._dag.subgraph(self._list_global_inds_unobserved)
        self._subset_data_arr = np.delete(
            self._data_arr,
            self._list_global_inds_unobserved, axis=1)

    @property
    def data(self):
        """
        return data in numpy array format
        """
        return self._subset_data_arr

    @property
    def mat_adj(self):
        """
        return adj matrix
        """
        return self._sub_dag.mat_adjacency

    def to_csv(self):
        """
        sub dataframe to  csv
        """
        node_names = [name for (i, name) in
                      enumerate(self._dag.list_node_names)
                      if i not in self._list_global_inds_unobserved]
        df = pd.DataFrame(self.data, columns=node_names)
        df.to_csv("output.csv", index=False)
        subdag = MatDAG(self.mat_adj)
        subdag.to_binary_csv()

    def visualize(self, title):
        """
        plot DAG
        """
        self._sub_dag.visualize(title=title)
