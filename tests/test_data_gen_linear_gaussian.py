"""
test linear gaussian data gen
"""
import pandas as pd
from causalSpyne.data_linear_gaussian import DataGenLinearGaussian
from causalSpyne.gen_dag_2level import GenDAG2Level
from causalSpyne.dag_gen import GenDAG


def test_data_gen_linear_gaussian():
    """
    test linear gaussian data gen
    """
    simple_dag_gen = GenDAG(num_nodes=3, degree=2, list_weight_range=[3, 5])
    dag_gen = GenDAG2Level(dag_generator=simple_dag_gen, num_macro_nodes=3)
    dag = dag_gen.run()
    gen_data = DataGenLinearGaussian(dag)
    arr = gen_data.gen(num_samples=2)
    arr.shape
    df = pd.DataFrame(arr,
                      columns=dag.list_node_names)
