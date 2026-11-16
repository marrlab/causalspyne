import numpy as np
from causalspyne.ancestral_acc import ancestral_auc, ancestral_relation_scores, total_effect_ancestor_scores, pag_ancestor_scores
from causalspyne.pair_shd import labels_admg, labels_estimate, labels_exact_mag, pair_shd

CHAIN = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])           # 0 -> 1 -> 2 as adj[i, j]: i -> j


def test_auc_perfect_and_reversed():
    S = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], float)
    assert ancestral_auc(CHAIN, S) == 1.0
    assert ancestral_auc(CHAIN, S.T) == 0.0
    assert ancestral_auc(CHAIN, np.ones((3, 3))) == 0.5


def test_total_effect_scores_rank_ancestors_first():
    B = np.array([[0, 0, 0], [0.7, 0, 0], [0, -0.5, 0]])       # B[i, j] != 0: j -> i, i.e. the chain 0 -> 1 -> 2
    assert ancestral_auc(CHAIN, total_effect_ancestor_scores(B)) == 1.0


def test_pag_scores():
    G = np.zeros((3, 3), int)
    G[0, 1], G[1, 0] = -1, 1      # 0 -> 1 definite
    G[1, 2], G[2, 1] = 2, 1       # 1 o-> 2 : 1 possible ancestor of 2
    S = pag_ancestor_scores(G)
    assert S[0, 1] == 1.0 and S[1, 2] == 0.5 and S[0, 2] == 0.5 and S[2, 0] == 0.0


def test_relation_scores_perfect():
    assert ancestral_relation_scores(CHAIN, CHAIN) == (1.0, 1.0, 1.0)


def test_pair_labels_and_shd():
    W = CHAIN.T.astype(float)                                   # W[child, parent]
    t = labels_admg(W, [(0, 2)])
    assert list(t) == [1, 3, 1]                                 # pairs (0,1), (0,2), (1,2)
    B = np.array([[0, 0, np.nan], [0.7, 0, 0], [np.nan, -0.5, 0]])
    assert pair_shd(t, labels_estimate(B)) == 0.0
    assert pair_shd(t, labels_estimate(np.zeros((3, 3)))) == 1.0


def test_typed_edge_shd_counts_bow_as_two_edges():
    from causalspyne.pair_shd import typed_edge_shd
    from causalspyne.ancestral_acc import ancestral_binary_accuracy
    d_true = np.array([[0, 1], [0, 0]]); b_true = np.array([[0, 1], [1, 0]])        # bow: 0 -> 1 and 0 <-> 1
    assert typed_edge_shd(d_true, b_true, d_true, b_true)["shd"] == 0
    assert typed_edge_shd(d_true, b_true, d_true, np.zeros((2, 2)))["shd"] == 1      # misses only the bidirected edge
    r = typed_edge_shd(d_true, b_true, d_true.T, np.zeros((2, 2)))                   # reversed edge: 1 FN + 1 FP, plus the missing <->
    assert r["shd"] == 3 and r["dir"] == {"fn": 1, "fp": 1}
    assert ancestral_binary_accuracy(d_true, d_true) == 1.0 and ancestral_binary_accuracy(d_true, d_true.T) == 0.0
