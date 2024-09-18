try:
    from contextlib import chdir
except Exception:
    from causalspyne.py3_9_10_compatibility import chdir

from pathlib import Path

from causalspyne.gen_dag_2level import GenDAG2Level
from causalspyne.dag_gen import GenDAG
from causalspyne.dag_viewer import DAGView


def gen_partially_observed(
    degree=2,
    list_confounder2hide=[0.5, 0.9],
    size_micro_node_dag=4,
    num_macro_nodes=4,
    num_sample=200,
    output_dir="output",
):
    """
    sole function as user interface
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    simple_dag_gen = GenDAG(num_nodes=size_micro_node_dag, degree=degree)

    # num_macro_nodes will overwrite behavior
    dag_gen = GenDAG2Level(
        dag_generator=simple_dag_gen, num_macro_nodes=num_macro_nodes
    )
    dag = dag_gen.run()
    with chdir(output_dir):
        dag.visualize(title="dag_complete")

    subview = DAGView(dag=dag)
    subview.run(
        num_samples=num_sample, confound=True, list_nodes2hide=list_confounder2hide
    )
    with chdir(output_dir):
        subview.to_csv()
    str_node2hide = "_".join(map(str, subview._list_nodes2hide))
    with chdir(output_dir):
        subview.visualize(title="dag_marginal_hide_" + str_node2hide)
        subview._sub_dag.to_binary_csv()
    return subview.data
