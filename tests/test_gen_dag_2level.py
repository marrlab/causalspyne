"""
test 2 level DAG generation
"""

from causalSpyne.gen_dag_2level import GenDAG2Level
from causalSpyne.dag_gen import GenDAG
from causalSpyne.is_dag import is_dag
from causalSpyne.draw_dags import draw_dags_nx

def test_gen_dag_2level():
    """
    test 2 level generation of DAGs
    """

    dag_gen = GenDAG(num_nodes=3, degree=2)
    gen = GenDAG2Level(dag_generator=dag_gen, num_macro_nodes=3)
    dag = gen.run()
    dag
    print(dag)
    is_dag(dag.mat_adjacency)
    draw_dags_nx(dag.mat_adjacency, dict_ind2name=dag.gen_dict_ind2node_na())
