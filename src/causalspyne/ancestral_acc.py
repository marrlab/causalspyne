"""
ancestral accuracy calculate the percentage of correct prediction
among all pairwise variables' ancestral relationships (binary)
"""
from itertools import combinations
import warnings

from causalspyne.dag2ancestral import DAG2Ancestral


def ancestral_acc(true_dag, pred_order, list_hidden_nodes=None):
    """
    Parameters:
    true_dag: Target causalspyne graph
    pred_order:
    ISSUE #34, the pairwise relationship can be + or - ancestral, but can also be no ancestral relationship
    list_hidden_nodes (list): list of nodes in DAG to hide, here we use pairwise combinations of pred_order, which
    already assumes the algorithm has a clear + order only. 
    Returns:
    float:
    """
    dag2ancestral = DAG2Ancestral(true_dag.mat_adjacency)
    dag2ancestral.pre_cal_n_hop()
    size_dag = true_dag.num_nodes
    if list_hidden_nodes is not None:
        n_vars = size_dag - len(list_hidden_nodes)
    else:
        n_vars = size_dag
    if n_vars != len(pred_order):
        warnings.warn(f"predicted causal order {pred_order} does not \
                      have the same number of observables!, hidden are: {list_hidden_nodes} \
                      now forcing number of variables to be the lengh of pred_order")
        n_vars = len(pred_order)

    pairwise_combinations = list(combinations(pred_order, 2))
 
    n_correct = 0
    for pair in pairwise_combinations:
        ancestor, offspring = pair
        if dag2ancestral.is_ancestor(ancestor, offspring):
            n_correct += 1
    num_combos = n_vars * (n_vars - 1) /2
    return float(n_correct) / num_combos


def ancestral_relation_scores(adj_true, adj_est_directed):
    """Accuracy of the estimated ancestral relations between observed variables.

    Parameters:
    adj_true: (n, n) array, adj_true[i, j] != 0 iff i -> j in the true DAG restricted to the observed variables
        (hidden root confounders do not create ancestral relations, so they are not needed; if hidden vertices lie on a
        directed path between observed ones, pass the transitive closure of the full DAG restricted to the observed ones).
    adj_est_directed: (n, n) array, adj_est_directed[i, j] != 0 iff the estimated graph has the *definite* directed edge i -> j.
        Bidirected, undirected and circle-marked edges are not counted as ancestral relations.
    Returns: (pair_acc, precision, recall):
        pair_acc: share of the n(n-1)/2 unordered pairs whose label (i ancestor of j, j ancestor of i, or no relation) is
            identical in the truth and in the estimate (transitive closure of the estimated directed edges);
        precision, recall: of the ordered pairs "i is an ancestor of j" (precision is 1.0 when nothing is claimed).
    """
    import numpy as np

    def closure(a):
        r = (np.asarray(a) != 0).astype(int)
        for _ in range(len(r)):
            r2 = ((r + r @ r) > 0).astype(int)
            if (r2 == r).all():
                break
            r = r2
        return r

    t, e = closure(adj_true), closure(adj_est_directed)
    iu = np.triu_indices(len(t), 1)
    lab = lambda m: np.where(m[iu] == 1, 1, np.where(m.T[iu] == 1, 2, 0))
    pair_acc = float(np.mean(lab(t) == lab(e)))
    tp = float(((t == 1) & (e == 1)).sum())
    precision = tp / e.sum() if e.sum() > 0 else 1.0
    recall = tp / t.sum() if t.sum() > 0 else 1.0
    return pair_acc, precision, recall
