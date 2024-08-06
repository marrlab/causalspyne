"""
create different views for the same DAG by hiding some variables
"""
import warnings
import numpy as np
import pandas as pd
from causalspyne.data_gen import DataGen
from causalspyne.dag_interface import MatDAG


def process_list2hide(list_ind_or_percentage, total_num):
    """
    list_ind_or_percentage can either be a list or a scalar float
    """
    if len(list_ind_or_percentage) > total_num:
        raise RuntimeError(f"there are {total_num} confounders to hide, less \
                           than the length of {list_ind_or_percentage}")

    list_ind = [min(int(ele * total_num), total_num - 1)
                if isinstance(ele, float) else ele
                for ele in list_ind_or_percentage]

    list_ind = list(set(list_ind))

    if max(list_ind) > total_num:
        raise RuntimeError(f"max value in {list_ind_or_percentage} is bigger \
                           than total number of variables {total_num} to hide")

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
        self._list_nodes2hide = None
        self._success = False

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
        if not self._dag.list_confounder:
            warnings.warn(f"there are no confounders in the graph {self._dag} \
                          !")
            return False
        list_toporder_confounder2hide = process_list2hide(
            list_toporder_confounder2hide, len(self._dag.list_confounder))

        list_ind_confounder_sorted = \
            [self._dag.list_ind_nodes_sorted.index(confounder)
             for confounder in self._dag.list_confounder]
        list_toporder_confounder_sub = \
            [list_ind_confounder_sorted[i]
             for i in list_toporder_confounder2hide]

        self.hide(list_toporder_confounder_sub)
        return True

    def hide(self, list_toporder_unobserved):
        """
        hide variables according to a list of global index of topological sort
        """
        list_toporder_unobserved = process_list2hide(
            list_toporder_unobserved, self._dag.num_nodes)

        # subset list
        self._list_nodes2hide = [self._dag.list_top_names[i]
                                 for i in list_toporder_unobserved]

        # FIXME: change to logger
        print("nodes to hide " + str(self._list_nodes2hide))

        self._list_global_inds_unobserved = \
            [self._dag.list_ind_nodes_sorted[ind_top_order]
             for ind_top_order in list_toporder_unobserved]

        self._sub_dag = self._dag.subgraph(self._list_global_inds_unobserved)
        self._subset_data_arr = np.delete(
            self._data_arr,
            self._list_global_inds_unobserved, axis=1)
        self._success = True

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
        if not self._success:
            warnings.warn("no subview of DAG available")
            return
        node_names = [name for (i, name) in
                      enumerate(self._dag.list_node_names)
                      if i not in self._list_global_inds_unobserved]
        df = pd.DataFrame(self.data, columns=node_names)
        str_node2hide = '_'.join(map(str, self._list_nodes2hide))
        df.to_csv(title + str_node2hide, index=False)
        subdag = MatDAG(self.mat_adj)
        subdag.to_binary_csv()

    def visualize(self, title):
        """
        plot DAG
        """
        if not self._success:
            warnings.warn("no subview of DAG available")
            return
        self._sub_dag.visualize(title=title)
