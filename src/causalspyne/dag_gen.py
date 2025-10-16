"""
concrete class to generate simple DAGs

"""

import warnings
from numpy.random import default_rng

from causalspyne.erdo_renyi_plp import Erdos_Renyi_PLP
from causalspyne.dag_interface import MatDAG
from causalspyne.weight import WeightGenWishart
from causalspyne.dag_manipulator import DAGManipulator


class GenDAG:
    def __init__(self, num_nodes, degree, obj_gen_weight=None,
                 rng=default_rng()):
        """
        degree: expected degree for each node
        """
        self.num_nodes = num_nodes
        self.degree = degree
        self.strategy_gen_dag = Erdos_Renyi_PLP(rng)
        self.obj_gen_weight = obj_gen_weight
        if obj_gen_weight is None:
            self.obj_gen_weight = WeightGenWishart(rng=rng)
        self.dag_manipulator = None
        self.rng = rng

    def gen_dag(self, num_nodes=None, prefix="", target_num_confounder=2):
        """
        generate DAG and wrap it around with interface
        """
        if num_nodes is None:
            num_nodes = self.num_nodes
        mat_skeleton = self.strategy_gen_dag(num_nodes, self.degree)

        mat_mask = (mat_skeleton != 0).astype(float)
        mat_weight = self.obj_gen_weight.gen(num_nodes)
        # Hardarmard product
        mat_weighted_adjacency = mat_mask * mat_weight

        dag = MatDAG(mat_weighted_adjacency, name_prefix=prefix, rng=self.rng)
        self.dag_manipulator = DAGManipulator(dag,
                                              self.obj_gen_weight, self.rng)
        ind_arbitrary = dag.get_top_last()
        counter = 0
        for _ in range(dag.num_nodes):
            flag_success = self.dag_manipulator.mk_confound(
                ind_arbitrary_confound_input=ind_arbitrary)
            if not flag_success:
                counter += 1
            if dag.num_confounder >= target_num_confounder:
                break
            # FIXME: it can be the new ind_arbitrary has been tried out already
            ind_arbitrary = dag.climb(ind_arbitrary)
            if ind_arbitrary is None:
                break
        num_confounder = len(dag.list_confounder)
        if num_confounder < target_num_confounder and \
                dag.num_nodes - target_num_confounder > 1:
            warnings.warn(
                f"\n failed to ensure {target_num_confounder} confounders for \
                adjacency matrix \n{dag.mat_adjacency}, \
                \n after {counter} failed trials, \
                \n{num_confounder} confounders only")
        return dag
