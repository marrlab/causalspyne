"""
test 2 level DAG generation
"""

from causalSpyne.gen_dag_2level import \
    GenDAG2Level, NumNodesPerCluster
from causalSpyne.dag_gen import GenDAG


def test_recursive_gen():

    dag_gen = GenDAG(num_nodes=3, degree=2, list_weight_range=[3, 5])
    getter_num_nodes = NumNodesPerCluster()
    gen = GenDAG2Level(dag_generator=dag_gen,
                       strategy_num_nodes_per_cluster=getter_num_nodes,
                       num_macro_nodes=3)
    gen.run()
