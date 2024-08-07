from causalspyne.gen_dag_2level import GenDAG2Level
from causalspyne.dag_gen import GenDAG
from causalspyne.dag_viewer import DAGView


def gen_partially_observed(degree=2,
                           list_confounder2hide=[0.5, 0.9],
                           size_micro_node_dag=4,
                           num_macro_nodes=4,
                           num_sample=200):
    """
    sole function as user interface
    """


    simple_dag_gen = GenDAG(num_nodes=size_micro_node_dag, degree=degree)

    # num_macro_nodes will overwrite behavior
    dag_gen = GenDAG2Level(dag_generator=simple_dag_gen, num_macro_nodes=num_macro_nodes)
    dag = dag_gen.run()
    dag.visualize(title="dag_complete")

    subview = DAGView(dag=dag)
    subview.run(num_samples=num_sample, confound=True, list_nodes2hide=list_confounder2hide)
    subview.to_csv()
    subview.visualize(title="dag_marginal")

    subview._sub_dag.to_binary_csv()
