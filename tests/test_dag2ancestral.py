"""
test data and DAG subview
"""
import numpy as np
from numpy.random import default_rng
from causalspyne.gen_dag_2level import GenDAG2Level
from causalspyne.dag_gen import GenDAG
from causalspyne.dag_viewer import DAGView
from causalspyne.dag2ancestral import DAG2Ancestral



def test_data_dag2ancestral():
    """
    """
    simple_dag_gen = GenDAG(num_nodes=3, degree=2, rng=default_rng(0))
    dag_gen = GenDAG2Level(
        dag_generator=simple_dag_gen, num_macro_nodes=3, rng=default_rng(0)
    )
    dag = dag_gen.run()
    dag.visualize(title="dag_complete")

    obj = DAG2Ancestral(dag.mat_adjacency)
    pred_ancestral_graph = obj.run([1, 2])
    print("\n")
    print(pred_ancestral_graph)
