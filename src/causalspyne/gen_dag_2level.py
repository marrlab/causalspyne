"""
2-level DAG generation
"""

from numpy.random import default_rng

from causalspyne.dag_stack_indexer import DAGStackIndexer
from causalspyne.dag_manipulator import DAGManipulator
from causalspyne.weight import WeightGenWishart


class GenDAG2Level:
    """
    generate a DAG with 2 levels: first level generate macro nodes, second
    level populate each macro node
    """

    def __init__(
        self, dag_generator, num_macro_nodes, max_num_local_nodes=4, rng=default_rng()
    ):
        self.dag_generator = dag_generator
        self.num_macro_nodes = num_macro_nodes
        self.max_num_local_nodes = max_num_local_nodes

        self.global_dag_indexer = None
        self.dag_backbone = None
        self.dict_macro_node2dag = {}
        self.dag_refined = None
        self.rng = rng
        self.dag_manipulator = None

    def populate_macro_node(self):
        """
        replace a macro node into a DAG
        """
        # iterate each macro node
        for name in self.dag_backbone.list_node_names:
            num_nodes = self.rng.integers(2, self.max_num_local_nodes + 1)
            self.dict_macro_node2dag[name] = self.dag_generator.gen_dag(
                num_nodes=num_nodes, prefix=name
            )
        self.global_dag_indexer = DAGStackIndexer(self)

    def interconnection(self):
        """
        connect macro nodes with edges
        """
        # iterate over the Macro-DAG edges
        for arc in self.dag_backbone.list_arcs:
            self.connect_macro_node_via_local_node(arc)

    def connect_macro_node_via_local_node(self, arc):
        """
        connect macro-DAG node edge (i,j) via local nodes
        """
        macro_arrow_tail, macro_arrow_head = arc
        _, ind_local_tail = self.dict_macro_node2dag[macro_arrow_tail].sample_node()
        _, ind_local_head = self.dict_macro_node2dag[macro_arrow_head].sample_node()

        ind_macro_tail = self.dag_backbone.get_node_ind(macro_arrow_tail)
        ind_macro_head = self.dag_backbone.get_node_ind(macro_arrow_head)

        ind_global_tail = self.global_dag_indexer.get_global_ind(
            ind_macro_tail, ind_local_tail
        )
        ind_global_head = self.global_dag_indexer.get_global_ind(
            ind_macro_head, ind_local_head
        )

        self.dag_refined.add_arc_ind(ind_global_tail, ind_global_head)

    def inject_additional_confounder(self):
        """
        make confounder in the big graph
        """
        obj_gen_weight = WeightGenWishart(rng=self.rng)
        self.dag_manipulator = DAGManipulator(self.dag_refined,
                                              obj_gen_weight, self.rng)

        ind_arbitrary = self.dag_refined.get_top_last()
        self.dag_manipulator.mk_confound(ind_arbitrary)
        print(self.dag_refined.num_confounder)
        ind_arbitrary = self.dag_refined.climb(ind_arbitrary)
        self.dag_manipulator.mk_confound(ind_arbitrary)
        print(self.dag_refined.num_confounder)

    def run(self):
        """
        generation
        """

        # generate dag_backbone DAG with only macro nodes
        self.dag_backbone = self.dag_generator.gen_dag(self.num_macro_nodes)
        self.populate_macro_node()
        self.interconnection()
        self.dag_refined.check()
        self.inject_additional_confounder()
        return self.dag_refined
