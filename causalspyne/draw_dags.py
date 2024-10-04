"""
draw DAG using networkx
"""

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def draw_dags_nx(
    adj_matrix, dict_ind2name=None, title="dag", ax=None, show=False, graphviz=False
):
    """
    networkx adjacency matrix (i,j) entry refers to edge from i pointing to j,
    which is opposite to the CausalSpyne convention
    """
    plt.close("all")
    nx_graph = nx.from_numpy_array(adj_matrix.transpose(), create_using=nx.DiGraph)
    if dict_ind2name:
        nx_graph = nx.relabel_nodes(nx_graph, dict_ind2name)
    if graphviz:
        pos = graphviz_layout(nx_graph, prog="dot")
    else:
        pos = nx.spring_layout(nx_graph, k=0.5, scale=2)
    nx.draw(nx_graph, pos=pos, ax=ax, arrows=True, with_labels=True, node_color="white")
    if ax is None:
        plt.title(title)
        plt.axis("off")
        plt.savefig(title + ".pdf", format="pdf")
        # plt.savefig(title + ".svg", format="svg")
    if show:
        plt.show()
