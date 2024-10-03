"""
generate DAG and its marginal DAG
"""

from datetime import datetime

try:
    from contextlib import chdir
except Exception:
    from causalspyne.py3_9_10_compatibility import chdir

from pathlib import Path

import matplotlib.pyplot as plt
from numpy.random import default_rng

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
    rng=default_rng(),
):
    """
    sole function as user interface
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("DAGs")

    simple_dag_gen = GenDAG(num_nodes=size_micro_node_dag, degree=degree, rng=rng)

    # num_macro_nodes will overwrite behavior
    dag_gen = GenDAG2Level(
        dag_generator=simple_dag_gen, num_macro_nodes=num_macro_nodes, rng=rng
    )
    dag = dag_gen.run()
    dag.visualize(title="complete", ax=ax1)
    ax1.set_title("complete")

    subview = DAGView(dag=dag, rng=rng)
    subview.run(
        num_samples=num_sample, confound=True, list_nodes2hide=list_confounder2hide
    )
    with chdir(output_dir):
        subview.to_csv()
    str_node2hide = subview.str_node2hide
    subview.visualize(title="marginal_hide_" + str_node2hide, ax=ax2)
    ax2.set_title("marginal_hide_" + str_node2hide)
    with chdir(output_dir):
        fig.savefig(f"{timestamp}dags.pdf", format="pdf")

    return subview.data
