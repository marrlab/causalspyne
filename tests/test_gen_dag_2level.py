"""
test 2 level DAG generation
"""

from numpy.random import default_rng

from causalspyne.gen_dag_2level import GenDAG2Level
from causalspyne.dag_gen import GenDAG
from causalspyne.is_dag import is_dag
from causalspyne.draw_dags import draw_dags_nx


def test_gen_dag_2level(tmp_path, monkeypatch):
    """
    test 2 level generation of DAGs
    """
    monkeypatch.chdir(tmp_path)

    dag_gen = GenDAG(num_nodes=3, degree=2, rng=default_rng(0))
    gen = GenDAG2Level(
        dag_generator=dag_gen,
        num_macro_nodes=3,
        num_micro_nodes=3,
        rng=default_rng(0),
    )
    dag = gen.run()
    dag
    print(dag)
    is_dag(dag.mat_adjacency)
    draw_dags_nx(dag.mat_adjacency, dict_ind2name=dag.gen_dict_ind2node_na())


def test_gen_dag_2level_random_micro_nodes_respects_max(tmp_path, monkeypatch):
    """
    test random local DAG sizes with configurable max local nodes
    """
    monkeypatch.chdir(tmp_path)

    dag_gen = GenDAG(num_nodes=5, degree=2, rng=default_rng(0))
    gen = GenDAG2Level(
        dag_generator=dag_gen,
        num_macro_nodes=3,
        num_micro_nodes=None,
        max_num_local_nodes=5,
        rng=default_rng(0),
    )
    gen.run()

    for local_dag in gen.dict_macro_node2dag.values():
        assert 2 <= local_dag.num_nodes <= 5
