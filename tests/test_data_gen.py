"""
test linear gaussian data gen
"""

from numpy.random import default_rng
import pandas as pd

from causalspyne.data_gen import DataGen
from causalspyne.gen_dag_2level import GenDAG2Level
from causalspyne.dag_gen import GenDAG


def test_data_gen_linear_gaussian():
    """
    test linear gaussian data gen
    """
    simple_dag_gen = GenDAG(num_nodes=3, degree=2, rng=default_rng(0))
    dag_gen = GenDAG2Level(
        dag_generator=simple_dag_gen, num_macro_nodes=2, rng=default_rng(0)
    )
    dag = dag_gen.run()
    gen_data = DataGen(dag, rng=default_rng(0))
    arr = gen_data.gen(num_samples=200)
    arr.shape
    df = pd.DataFrame(arr, columns=dag.list_node_names)
    df.to_csv("output.csv", index=False)
    dag.to_binary_csv()
