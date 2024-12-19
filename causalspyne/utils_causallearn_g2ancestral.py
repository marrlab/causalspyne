"""
A graph is simpler than simplicial complex in that it only characterize
pairwise relationship, which enables us to project a complicated graph in
causallearn to ancestral ADMG (i.e. ancestral graph with only directed edges)

We of course lose information via projection.

This module offers another function to extract causal order from
causal-learn
"""
from causallearn.graph.NodeType import NodeType


def project_causallearn_g2ancestral_admg(mat_graph_causallearn):
    """
    A general graph in causal-learn is not necessarily a PAG (it does have
    an attribute to denote whether it is a PAG or not)
    When it is a PAG, like the return from FCI:
    https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/FCI.html#usage
    g.graph is a PAG
    A->B:   G[A, B]=-1, G[B, A]=1, to our format, G[A, B]=0, G[B, A]=1
    A.->B:  G[A, B]=2, G[B, A]=1, B is not an ancestor of A,
    to our format, G[A, B]=0, G[B, A]=1
    A.-.B:  G[A, B]=2, G[B,A]=2, no set d-separates A and B, ?????
    A<->B: there is common latent cause, G[A, B]=1, G[B, A]=1,
    no need to convert
    """
    for i in range(len(mat_graph_causallearn)):
        for j in range(len(mat_graph_causallearn[i])):
            if mat_graph_causallearn[i][j] != 1:
                mat_graph_causallearn[i][j] = 0
    return mat_graph_causallearn


def get_causal_order(g_causal_learn, node_names=None):
    """
    get causal order
    """
    real_name_order = []
    real_na_order_latent = []
    nodes_order = g_causal_learn.get_causal_ordering()
    for node in nodes_order:
        if node.get_node_type() == NodeType.LATENT:
            # GIN-LINLAM
            real_na_order_latent.append(node.get_node_type)
            continue
        fake_name = node.get_name()
        index = int(fake_name.removeprefix("X")) - 1
        real_name = node_names[index]
        print(f"{real_name}")
        real_name_order.append(real_name)
        real_na_order_latent.append(real_name)
    print(f"{real_name_order}")
    return real_name_order, real_na_order_latent
