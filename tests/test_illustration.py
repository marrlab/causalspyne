"""tests for the confounding illustration (graphs and block matrices)"""
import numpy as np
import matplotlib

matplotlib.use("Agg")

from causalspyne.illustration import ConfoundingIllustration, layered_positions

# example DAG of the CLeaR 2027 paper, edges parent -> child
EDGES = [(1, 0), (1, 2), (1, 5), (2, 0), (2, 4), (5, 3), (5, 4), (6, 3),
         (6, 4), (0, 4), (3, 4)]


def paper_dag():
    adj = np.zeros((7, 7), dtype=int)
    for parent, child in EDGES:
        adj[child, parent] = 1
    return adj


def test_hidden_root_gives_two_bidirected_edges():
    ill = ConfoundingIllustration(paper_dag(), hidden=[1])
    assert sorted(ill.bidirected) == [(0, 5), (2, 5)]
    assert sorted(ill.directed) == [(0, 4), (2, 0), (2, 4), (3, 4), (5, 3),
                                    (5, 4), (6, 3), (6, 4)]


def test_hidden_intermediate_gives_no_bidirected_edge():
    ill = ConfoundingIllustration(paper_dag(), hidden=[2, 5])
    assert ill.bidirected == []
    assert sorted(ill.directed) == [(0, 4), (1, 0), (1, 3), (1, 4), (3, 4),
                                    (6, 3), (6, 4)]


def test_block_structure_of_explicit_matrix():
    ill = ConfoundingIllustration(paper_dag(), hidden=[1])
    mat = ill.explicit_matrix()
    n_obs = len(ill.observed)
    assert mat.shape == (7, 7)
    # hidden root: W_UO = 0 and W_UU = 0
    assert not mat[n_obs:, :].any()
    # W_OU = Lambda has one nonzero entry per child of the hidden vertex
    assert mat[:n_obs, n_obs:].sum() == 3
    ill2 = ConfoundingIllustration(paper_dag(), hidden=[2, 5])
    mat2 = ill2.explicit_matrix()
    assert mat2[len(ill2.observed):, :len(ill2.observed)].sum() == 2  # W_UO != 0


def test_omega_pattern_matches_bidirected_edges():
    ill = ConfoundingIllustration(paper_dag(), hidden=[1])
    pattern = ill.omega_pattern()
    assert (pattern == pattern.T).all()
    assert pattern.sum() == len(ill.observed) + 2 * len(ill.bidirected)
    diag = ConfoundingIllustration(paper_dag(), hidden=[2, 5]).omega_pattern()
    assert (diag == np.eye(5)).all()


def test_layout_and_outputs():
    adj = paper_dag()
    pos = layered_positions(adj)
    assert set(pos) == set(range(7))
    ill = ConfoundingIllustration(adj, hidden=[2, 5])
    assert ill.to_matplotlib() is not None
    tex = ill.to_tikz()
    assert tex.count("\\begin{tikzpicture}") == 3
    assert "W_{OU}" in tex and "\\node[hid]" in tex
