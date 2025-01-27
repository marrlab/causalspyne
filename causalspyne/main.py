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

from causalspyne.draw_dags import draw_dags_nx


def gen_partially_observed(
    degree=2,
    list_confounder2hide=[0.5, 0.9],
    size_micro_node_dag=4,
    num_macro_nodes=4,
    num_sample=200,
    output_dir="output/",
    rng=default_rng(),
    graphviz=False,
    plot=True,
):
    """
    sole function as user interface
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    simple_dag_gen = GenDAG(num_nodes=size_micro_node_dag,
                            degree=degree, rng=rng)

    # num_macro_nodes will overwrite behavior
    dag_gen = GenDAG2Level(
        dag_generator=simple_dag_gen,
        num_macro_nodes=num_macro_nodes, rng=rng
    )
    dag = dag_gen.run()
    dag.to_binary_csv(benchpress=False,
                      name=output_dir + "ground_truth_dag.csv")

    subview = DAGView(dag=dag, rng=rng)
    subview.run(
        num_samples=num_sample, confound=True,
        list_nodes2hide=list_confounder2hide
    )
    with chdir(output_dir):
        subview.to_csv()
    str_node2hide = subview.str_node2hide

    dag2ancestral = DAG2Ancestral(dag.mat_adjacency)
    list_confounder2hide_global_ind = subview.list_global_inds_nodes2hide
    pred_ancestral_graph_mat = dag2ancestral.run(
        list_confounder2hide_global_ind)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        mtitle = "hide_" + str_node2hide
        fig.suptitle(mtitle)  # super-title

        # ax1
        dag.visualize(title="DAG", ax=ax1, graphviz=graphviz)
        ax1.set_title("DAG")

        # ax2
        draw_dags_nx(
            pred_ancestral_graph_mat,
            dict_ind2name={
                i: name for i, name in enumerate(sorted(subview.node_names))
            },
            title="ancestral",
            ax=ax2,
            graphviz=graphviz,
        )
        ax2.set_title("ancestral")
        # ax3
        subview.visualize(
            title="subDAG", ax=ax3, graphviz=graphviz
        )
        ax3.set_title("subDAG")

    with chdir(output_dir):
        subview.to_csv()
        if plot:
            fig.savefig(f"graph_compare_{timestamp}dags.pdf", format="pdf")
            fig.savefig(f"graph_compare_{timestamp}dags.svg", format="svg")
        with open("hidden_nodes.csv", "w") as outfile:
            outfile.write(
                ",".join(str(node) for node in
                         subview._list_global_inds_unobserved)
            )
    subview_global_inds = [dag._dict_node_names2ind[name]
                           for name in dag.list_node_names if
                           name not in str_node2hide]
    return subview, subview.node_names, dag, subview_global_inds


def ordered_ind_col2global_ind(inds_cols, subview_global_inds):
    """
    given a predicted causal order in the form of column indices, transform it
    into global index of ground truth DAG
    """
    list_global_inds = [subview_global_inds[ind_col] for ind_col in inds_cols]
    return list_global_inds
