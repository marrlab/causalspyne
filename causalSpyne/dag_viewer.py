"""
create different views for the same DAG by hiding some variables
"""
import numpy as np
import pandas as pd
from causalSpyne.data_gen import DataGen
from causalSpyne.dag_interface import MatDAG


def process_list2hide(list_ind_or_percentage, total_num):
    """
    list_ind_or_percentage can either be a list or a scalar float
    """
    if len(list_ind_or_percentage) > total_num:
        raise RuntimeError(f"there are less confounders {total_num} to hide \
                           than the length of {list_ind_or_percentage}")

    if max(list_ind_or_percentage) > total_num:
        raise RuntimeError(f"max value in {list_ind_or_percentage} is bigger \
                           than total number of variables {total_num} to hide")

    list_ind = [int(ele * total_num) if isinstance(ele, float) else ele
                for ele in list_ind_or_percentage]
    return list_ind


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
        self._nodes2hide = None

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

    def hide_confounder(self, list_toporder_confounder2hide):
        """
        given a list of index, hide the confounder according to the toplogical
        order provided by the input index list_toporder_confounder2hide
        then call self.hide
        """
        list_toporder_confounder2hide = process_list2hide(
            list_toporder_confounder2hide, len(self._dag.list_confounder))

        list_ind_confounder_sorted = \
            [self._dag.list_ind_nodes_sorted.index(confounder)
             for confounder in self._dag.list_confounder]

        list_toporder_confounder_sub = \
            [list_ind_confounder_sorted[i]
             for i in list_toporder_confounder2hide]

        self.hide(list_toporder_confounder_sub)

    def hide(self, list_toporder_unobserved):
        """
        hide variables according to a list of global index of topological sort
        """
        list_toporder_unobserved = process_list2hide(
            list_toporder_unobserved, self._dag.num_nodes)

        # subset list
        self._nodes2hide = [self._dag.list_top_names[i]
                            for i in list_toporder_unobserved]
        # FIXME: change to logger
        print("nodes to hide " + str(self._nodes2hide))

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

    def to_csv(self, title="data_subdag.csv"):
        """
        sub dataframe to  csv
        """
        node_names = [name for (i, name) in
                      enumerate(self._dag.list_node_names)
                      if i not in self._list_global_inds_unobserved]
        df = pd.DataFrame(self.data, columns=node_names)
        df.to_csv(title, index=False)
        subdag = MatDAG(self.mat_adj)
        subdag.to_binary_csv()

    def visualize(self, title):
        """
        plot DAG
        """
        self._sub_dag.visualize(title=title)
