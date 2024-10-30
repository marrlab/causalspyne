"""
create different views for the same DAG by hiding some variables
"""

import warnings

import numpy as np
from numpy.random import default_rng
import pandas as pd

from causalspyne.data_gen import DataGen
from causalspyne.dag_interface import MatDAG


def process_list2hide(list_ind_or_percentage, total_num):
    """
    list_ind_or_percentage can either be a list or a scalar float
    """
    if len(list_ind_or_percentage) > total_num:
        raise RuntimeError(
            f"there are {total_num} confounders to hide, less \
                           than the length of {list_ind_or_percentage}"
        )

    list_ind = [
        min(int(ele * total_num), total_num - 1)
        if isinstance(ele, float) else ele
        for ele in list_ind_or_percentage
    ]

    list_ind = list(set(list_ind))

    if max(list_ind) > total_num:
        raise RuntimeError(
            f"max value in {list_ind_or_percentage} is bigger \
                           than total number of variables {total_num} to hide"
        )

    return list_ind


class DAGView:
    """
    with ground truth DAG intact, only show subgraph
    """

    def __init__(self, dag, rng=default_rng(0)):
        self._dag = dag
        # there is no need to use a full DAG to represent subdag
        # since sub-dag is not responsible for data generation
        self._sub_dag = None
        self._data_arr = None
        self._subset_data_arr = None
        self._list_global_inds_unobserved = None
        self._list_global_inds_observed = None
        self.data_gen = DataGen(self._dag, rng=rng)
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
            self.hide_top_order(list_nodes2hide)

    def hide_confounder(self, list_toporder_confounder2hide_input):
        """
        given a list of index, hide the confounder according to the toplogical
        order provided by the input index list_toporder_confounder2hide
        then call self.hide
        """
        if not self._dag.list_confounder:
            raise RuntimeError(
                f"there are no confounders in the graph {self._dag} \
                          !"
            )

        list_toporder_confounder2hide = process_list2hide(
            list_toporder_confounder2hide_input, len(self._dag.list_confounder)
        )

        list_ind_confounder_sorted = self._dag.list_top_order_sorted_confounder

        list_toporder_confounder_sub = [
            list_ind_confounder_sorted[i]
            for i in list_toporder_confounder2hide
        ]

        self.hide_top_order(list_toporder_confounder_sub)
        return True

    def hide_top_order(self, list_toporder_unobserved):
        """
        hide variables according to a list of global index of topological sort
        """
        list_toporder_unobserved = process_list2hide(
            list_toporder_unobserved, self._dag.num_nodes
        )

        # subset list
        self._list_nodes2hide = [
            self._dag.list_top_names[i] for i in list_toporder_unobserved
        ]

        # FIXME: change to logger
        print("nodes to hide " + str(self._list_nodes2hide))

        self._list_global_inds_unobserved = [
            self._dag.list_ind_nodes_sorted[ind_top_order]
            for ind_top_order in list_toporder_unobserved
        ]

        self._sub_dag = self._dag.subgraph(self._list_global_inds_unobserved)
        self._subset_data_arr = np.delete(
            self._data_arr, self._list_global_inds_unobserved, axis=1
        )
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

    def check_if_subview_done(self):
        """
        check if DAG marginalizatino successfull or not
        """
        if not self._success:
            warnings.warn("no subview of DAG available, exit now!")
            return

    @property
    def node_names(self):
        # filter out observed variable
        # _node_names = [name for (i, name) in
        #               enumerate(self._dag.list_node_names)
        #               if i not in self._list_global_inds_unobserved]
        _node_names = self._sub_dag.list_node_names
        _node_names_ind = ["X" + str(self._sub_dag._parent_list_node_names.index(name)) for
        name in _node_names]
        return _node_names_ind

    def to_csv(self, title="data_subdag.csv"):
        """
        sub dataframe to  csv
        """
        self.check_if_subview_done()

        # FIXME: ensure self.node_names are consistent with self.data
        df = pd.DataFrame(self.data, columns=self.node_names)
        df.to_csv(title[:-4] + "_" + self.str_node2hide + title[-4:],
                  index=False)

        subdag = MatDAG(self.mat_adj)
        subdag.to_binary_csv()

    @property
    def list_global_inds_nodes2hide(self):
        return self._list_global_inds_unobserved

    @property
    def list_global_inds_observed(self):
        if self._list_global_inds_observed is None:
            if self._list_global_inds_unobserved is None:
                raise RuntimeError(
                    "global inds for unobserved not initialized yet!")
            self._list_global_inds_observed = \
                [item for item in self._dag.list_ind_nodes_sorted
                 if item not in self._list_global_inds_unobserved]
        return self._list_global_inds_observed

    @property
    def str_node2hide(self):
        """
        string representation of nodes to hide(marginalize)
        """
        self.check_if_subview_done()
        if self._list_nodes2hide is None:
            raise RuntimeError("self._list_node2hide is None!")
        _str_node2hide = "_".join(map(str, self._list_nodes2hide))

        _str_node2hide_ind = "_".join(map(str, self._list_global_inds_unobserved))

        return "_ind_".join([_str_node2hide, _str_node2hide_ind])

    def visualize(self, **kwargs):
        """
        plot DAG
        """
        if not self._success:
            warnings.warn("no subview of DAG available")
            return
        self._sub_dag.visualize(**kwargs)
