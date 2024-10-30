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
from causalspyne.dag2ancestral import DAG2Ancestral

from causalspyne.dag_interface import MatDAG
from causalspyne.draw_dags import draw_dags_nx


def gen_partially_observed(
    degree=2,
    list_confounder2hide=[0.5, 0.9],
    size_micro_node_dag=4,
    num_macro_nodes=4,
    num_sample=200,
    output_dir="output",
    rng=default_rng(),
    graphviz=False,
):
    """
    sole function as user interface
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("graph comparison")  # super-title

    simple_dag_gen = GenDAG(num_nodes=size_micro_node_dag, degree=degree, rng=rng)

    # num_macro_nodes will overwrite behavior
    dag_gen = GenDAG2Level(
        dag_generator=simple_dag_gen, num_macro_nodes=num_macro_nodes, rng=rng
    )
    dag = dag_gen.run()
    dag.visualize(title="complete", ax=ax1, graphviz=graphviz)
    ax1.set_title("complete")

    subview = DAGView(dag=dag, rng=rng)
    subview.run(
        num_samples=num_sample, confound=True, list_nodes2hide=list_confounder2hide
    )
    with chdir(output_dir):
        subview.to_csv()
    str_node2hide = subview.str_node2hide

    dag2ancestral = DAG2Ancestral(dag.mat_adjacency)
    list_confounder2hide_global_ind = subview.list_global_inds_nodes2hide
    pred_ancestral_graph_mat = dag2ancestral.run(list_confounder2hide_global_ind)

    draw_dags_nx(
        pred_ancestral_graph_mat,
        dict_ind2name={i: name for i, name in enumerate(sorted(subview.node_names))},
        title="ancestral",
        ax=ax2,
        graphviz=graphviz,
    )
    ax2.set_title("hide_" + str_node2hide)

    subview.visualize(title="marginal_hide_" + str_node2hide, ax=ax3, graphviz=graphviz)
    ax3.set_title("")

    with chdir(output_dir):
        # subview.visualize(
        #    title="dag_marginal_hide_" + timestamp + str_node2hide)
        subview.to_csv()
        fig.savefig(f"{timestamp}dags.pdf", format="pdf")
        with open("hidden_nodes.csv", "w") as outfile:
            outfile.write(
                ",".join(str(node) for node in subview._list_global_inds_unobserved)
            )
    return subview.data, subview.node_names
