"""Pair-level structural Hamming distance for estimates of linear SEMs with hidden confounders (used by the CLeaR 2027 experiments).

Every unordered pair {i, j} of observed variables gets one of four labels: 0 no edge, 1 i -> j, 2 j -> i, 3 i <-> j (i < j).
The pair SHD is the share of the n(n-1)/2 pairs with a different label in the truth and in the estimate. Truth can be the ADMG
(directed edges of W and the confounded pairs) or the exact MAG of the DAG that contains one hidden root per confounded pair.
Convention: W[child, parent] != 0 means parent -> child; estimates B[i, j] != 0 means j -> i, NaN marks a confounded pair.
"""
import numpy as np


def _upper(m):
    return m[np.triu_indices(len(m), 1)]


def labels_admg(W, pairs):
    """Labels of the ADMG: directed edges of W, bidirected edges for the confounded pairs (no bow in the generators used so far)."""
    n = len(W)
    lab = np.zeros((n, n), int)
    conf = np.zeros((n, n), bool)
    for i, j in pairs:
        conf[i, j] = conf[j, i] = True
    for i in range(n):
        for j in range(i + 1, n):
            lab[i, j] = 3 if conf[i, j] else 1 if W[j, i] else 2 if W[i, j] else 0
    return _upper(lab)


def labels_estimate(B):
    """Labels of an estimated weight matrix B (NaN in B[i, j] or B[j, i]: confounded pair)."""
    n = len(B)
    lab = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(B[i, j]) or np.isnan(B[j, i]):
                lab[i, j] = 3
            else:
                lab[i, j] = 1 if B[j, i] != 0 else 2 if B[i, j] != 0 else 0
    return _upper(lab)


def exact_pag(W, pairs):
    """Exact PAG (causal-learn encoding, G[a, b] mark at a) of the DAG with edges of W and one hidden root per confounded pair."""
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.DAG2PAG import dag2pag
    n, k = len(W), len(pairs)
    nodes = [GraphNode(f"X{i}") for i in range(n + k)]
    g = Dag(nodes)
    for ch, pa in np.argwhere(W != 0):
        g.add_directed_edge(nodes[pa], nodes[ch])
    for l, (i, j) in enumerate(pairs):
        g.add_directed_edge(nodes[n + l], nodes[i])
        g.add_directed_edge(nodes[n + l], nodes[j])
    pag = dag2pag(g, nodes[n:])
    idx = [int(v.get_name()[1:]) for v in pag.get_nodes()]   # dag2pag scrambles the node order
    T = np.zeros((n, n), int)
    T[np.ix_(idx, idx)] = pag.graph
    return T


def labels_exact_mag(W, pairs, pag=None):
    """Labels of the exact MAG: adjacent in the exact PAG; i -> j if i is an ancestor of j in the observed part of the DAG, else i <-> j."""
    from causalspyne.ancestral_acc import ancestor_closure
    T = exact_pag(W, pairs) if pag is None else pag
    reach = ancestor_closure((np.asarray(W).T != 0).astype(int))
    n = len(W)
    lab = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            if T[i, j] != 0:
                lab[i, j] = 1 if reach[i, j] else 2 if reach[j, i] else 3
    return _upper(lab)


def pair_shd(true_labels, est_labels):
    """Share of the pairs whose labels differ."""
    return float(np.mean(np.asarray(true_labels) != np.asarray(est_labels)))


def typed_edge_shd(dir_true, bi_true, dir_est, bi_est):
    """SHD that treats the edges of the three types separately, so that bows are two edges and not one pair label:
    false negatives (edge of the type in the truth, not in the estimate) plus false positives (edge of the type in the estimate,
    not in the truth), summed over the types i -> j (directed, ordered pairs, so -> and <- are different edges) and i <-> j
    (bidirected, unordered pairs).

    dir_true, dir_est: adjacency matrices with [i, j] != 0 iff i -> j; bi_true, bi_est: symmetric matrices, nonzero off the
    diagonal iff i <-> j. Returns a dict with ``fn`` and ``fp`` per type ("dir", "bi") and the total ``shd``."""
    def mask(m):
        m = (np.nan_to_num(np.asarray(m, float)) != 0)
        np.fill_diagonal(m, False)
        return m
    dt, de, bt, be = mask(dir_true), mask(dir_est), mask(bi_true), mask(bi_est)
    bt, be = np.triu(bt | bt.T, 1), np.triu(be | be.T, 1)
    out = {"dir": {"fn": int((dt & ~de).sum()), "fp": int((de & ~dt).sum())},
           "bi": {"fn": int((bt & ~be).sum()), "fp": int((be & ~bt).sum())}}
    out["shd"] = sum(v for t in ("dir", "bi") for v in out[t].values())
    return out
