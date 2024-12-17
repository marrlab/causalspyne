
def utils_causallearn_g2ancestral():
    """
    https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/FCI.html#usage
    g.graph is a PAG
    A->B:   G[A, B]=-1, G[B, A]=1, to our format, G[A, B]=0, G[B, A]=1
    A.->B:  G[A, B]=2, G[B, A]=1, B is not an ancestor of A,
    to our format, G[A, B]=0, G[B, A]=1
    A.-.B:  G[A, B]=2, G[B,A]=2, no set d-separates A and B, ?????
    A<->B: there is common latent cause, G[A, B]=1, G[B, A]=1, no need to convert
    """

    self.mat4ancestral[c2_global_ind, c1_global_ind] = 1
    dag2ancestral = DAG2Ancestral(dag.mat_adjacency)
    list_confounder2hide_global_ind = subview.list_global_inds_nodes2hide
    pred_ancestral_graph_mat = dag2ancestral.run(
        list_confounder2hide_global_ind)
