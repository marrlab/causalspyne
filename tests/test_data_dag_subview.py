"""
test data and DAG subview
"""
import pandas as pd
from causalSpyne.data_linear_gaussian import DataGen
from causalSpyne.gen_dag_2level import GenDAG2Level
from causalSpyne.dag_gen import GenDAG
from causalSpyne.dag_viewer import DAGView


def test_data_dag_subview():
    """
    test linear gaussian data gen
    """
    simple_dag_gen = GenDAG(num_nodes=3, degree=2, list_weight_range=[3, 5])
    dag_gen = GenDAG2Level(dag_generator=simple_dag_gen, num_macro_nodes=2)
    dag = dag_gen.run()

    subview = DAGView(dag=dag)
    subview.run(num_samples=200)
    subview.to_csv()
