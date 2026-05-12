
"""
test data and DAG subview
"""

from numpy.random import default_rng

from causalspyne.gen_dag_2level import GenDAG2Level
from causalspyne.dag_gen import GenDAG
from causalspyne.dag_viewer import DAGView


def test_data_dag_subview(tmp_path, monkeypatch):
    """
    test linear gaussian data gen
    """
    monkeypatch.chdir(tmp_path)
    simple_dag_gen = GenDAG(num_nodes=3, degree=2, rng=default_rng(0))
    dag_gen = GenDAG2Level(
        dag_generator=simple_dag_gen, num_macro_nodes=2, rng=default_rng(0)
    )
    dag = dag_gen.run()
    dag.visualize(title="dag_complete")
    subview = DAGView(dag=dag, dft_noise="Gaussian")
    # subview.run(num_samples=200)
    subview.run(num_samples=200, list_nodes2hide=[1, 2])
    subview.to_csv()
    subview.visualize(title="dag_marginal")


def test_data_dag_subview_confounder(tmp_path, monkeypatch):
    """
    test linear gaussian data gen
    """
    monkeypatch.chdir(tmp_path)
    simple_dag_gen = GenDAG(num_nodes=3, degree=2, rng=default_rng(0))
    dag_gen = GenDAG2Level(
        dag_generator=simple_dag_gen, num_macro_nodes=2, rng=default_rng(0)
    )
    dag = dag_gen.run()
    dag.visualize(title="dag_complete_confounder")

    subview = DAGView(dag=dag,dft_noise="Gaussian")
    subview.run(num_samples=200, confound=True, list_nodes2hide=[0])
    subview.to_csv()
    subview.visualize(title="dag_marginal_confounder")


def test_data_dag_subview_confounder_percentage_single(tmp_path, monkeypatch):
    """
    test linear gaussian data gen
    """
    monkeypatch.chdir(tmp_path)
    simple_dag_gen = GenDAG(num_nodes=4, degree=2, rng=default_rng(0))
    dag_gen = GenDAG2Level(
        dag_generator=simple_dag_gen, num_macro_nodes=4, rng=default_rng(0)
    )
    dag = dag_gen.run()
    dag.visualize(title="dag_complete_confounder")

    subview = DAGView(dag=dag,dft_noise="Gaussian")
    subview.run(num_samples=200, confound=True, list_nodes2hide=[0.99])
    subview.to_csv()
    subview.visualize(title="dag_marginal_confounder")


def test_data_dag_subview_confounder_percentage_multiple(tmp_path, monkeypatch):
    """
    test linear gaussian data gen
    """
    monkeypatch.chdir(tmp_path)
    simple_dag_gen = GenDAG(num_nodes=4, degree=2, rng=default_rng(0))
    dag_gen = GenDAG2Level(
        dag_generator=simple_dag_gen, num_macro_nodes=4, rng=default_rng(0)
    )
    dag = dag_gen.run()
    dag.visualize(title="dag_complete_confounder2")

    subview = DAGView(dag=dag,dft_noise="Gaussian")
    subview.run(num_samples=200, confound=True, list_nodes2hide=[0.1, 1.0])
    subview.to_csv()
    subview.visualize(title="dag_marginal_confounder2")
