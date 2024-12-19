"""
A graph is simpler than simplicial complex in that it only characterize
pairwise relationship, which enables us to project a complicated graph in
causallearn to ancestral ADMG (i.e. ancestral graph with only directed edges)

We of course lose information via projection.
"""


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
