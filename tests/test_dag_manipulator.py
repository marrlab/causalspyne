"""
test linear gaussian data gen
"""
from causalSpyne.dag_gen import GenDAG
# from causalSpyne.dag_manipulator import DAGManipulator
from causalSpyne.gen_dag_2level import GenDAG2Level


def test_data_gen_linear_gaussian():
    """
    test linear gaussian data gen
    """
    simple_dag_gen = GenDAG(num_nodes=30, degree=2, list_weight_range=[3, 5])
    dag = simple_dag_gen.gen_dag()
    # DAGManipulator(dag)


def test_data_dag_subview():
    """
    test linear gaussian data gen
    """
    simple_dag_gen = GenDAG(num_nodes=3, degree=2, list_weight_range=[3, 5])
    dag_gen = GenDAG2Level(dag_generator=simple_dag_gen, num_macro_nodes=2)
    dag = dag_gen.run()
    dag.visualize(title="dag_complete_test")