"""
test 2 level DAG generation
"""

from numpy.random import default_rng

from causalspyne.gen_dag_2level import GenDAG2Level
from causalspyne.dag_gen import GenDAG
from causalspyne.is_dag import is_dag
from causalspyne.draw_dags import draw_dags_nx


def test_gen_dag_2level():
    """
    test 2 level generation of DAGs
    """

    dag_gen = GenDAG(num_nodes=3, degree=2, rng=default_rng(0))
    gen = GenDAG2Level(dag_generator=dag_gen, num_macro_nodes=3, rng=default_rng(0))
    dag = gen.run()
    dag
    print(dag)
    is_dag(dag.mat_adjacency)
    draw_dags_nx(dag.mat_adjacency, dict_ind2name=dag.gen_dict_ind2node_na())
