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


def ancestor_closure(adj):
    """Transitive closure: out[i, j] = 1 iff there is a directed path i -> ... -> j (i != j) in the graph with adj[i, j] != 0 for i -> j."""
    import numpy as np
    r = (np.asarray(adj) != 0).astype(int)
    np.fill_diagonal(r, 0)
    for _ in range(len(r)):
        r2 = ((r + r @ r) > 0).astype(int)
        if (r2 == r).all():
            break
        r = r2
    np.fill_diagonal(r, 0)
    return r


def ancestral_auc(adj_true, score):
    """Area under the ROC curve of the ranking of the ordered pairs (i, j), i != j, by ``score[i, j]`` ("i is an ancestor of j").

    adj_true[i, j] != 0 iff i -> j in the true DAG restricted to the observed variables (its transitive closure gives the
    positive pairs). Ties get the average rank. Returns nan if there is no positive or no negative pair.
    """
    import numpy as np
    from scipy.stats import rankdata
    truth = ancestor_closure(adj_true)
    off = ~np.eye(len(truth), dtype=bool)
    y, s = truth[off], np.asarray(score, float)[off]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def total_effect_ancestor_scores(B):
    """Ancestor scores from an estimated weight matrix B (B[i, j] != 0: j -> i; NaN entries, e.g. confounded pairs, count as 0):
    score[i, j] = |[(I - B)^-1]_{j, i}|, the absolute total effect of i on j."""
    import numpy as np
    B = np.nan_to_num(np.asarray(B, float))
    T = np.linalg.inv(np.eye(len(B)) - B)
    S = np.abs(T).T
    np.fill_diagonal(S, 0.0)
    return S


def pag_ancestor_scores(G):
    """Ancestor scores from a PAG in the causal-learn encoding (G[a, b] is the mark at a of the edge a-b: -1 tail, 1 arrowhead,
    2 circle; G[a, b] = 0: no edge): 1.0 if i is a definite ancestor of j (directed path of definite edges i -> j), 0.5 if i is a
    possible ancestor (a path whose edges all can be directed away from i: tail or circle at the tail side, arrowhead or circle at
    the head side) and 0 otherwise."""
    import numpy as np
    G = np.asarray(G)
    n = len(G)
    definite = np.zeros((n, n), int)
    possible = np.zeros((n, n), int)
    for a in range(n):
        for b in range(n):
            if a != b and G[a, b] != 0 and G[b, a] != 0:
                if G[a, b] == -1 and G[b, a] == 1:
                    definite[a, b] = 1
                if G[a, b] in (-1, 2) and G[b, a] in (1, 2):
                    possible[a, b] = 1
    return ancestor_closure(definite) + 0.5 * (ancestor_closure(possible) * (1 - ancestor_closure(definite)))


def ancestral_binary_accuracy(adj_true, adj_est_directed):
    """(True positives + true negatives) / all ordered pairs (u, v), u != v, for the question "is u an ancestor of v"
    (the measure used for ancestral relations by Wang and Drton, JMLR 2023, Sec. 6.1). Both arguments are adjacency matrices of
    directed edges (adj[i, j] != 0: i -> j); the estimated ancestors are the transitive closure of the estimated directed edges."""
    import numpy as np
    t, e = ancestor_closure(adj_true), ancestor_closure(adj_est_directed)
    off = ~np.eye(len(t), dtype=bool)
    return float(np.mean(t[off] == e[off]))
