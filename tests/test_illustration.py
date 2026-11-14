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


def test_omega_pattern_follows_the_explicit_model():
    # hidden root 1 loads on 0, 2, 5: Omega has all three off-diagonal pairs; 0<->2 is a bow (2 -> 0)
    ill = ConfoundingIllustration(paper_dag(), hidden=[1])
    pattern = ill.omega_pattern()
    pos = {v: k for k, v in enumerate(ill.observed)}
    assert (pattern == pattern.T).all()
    assert pattern[pos[0], pos[5]] == 1 and pattern[pos[2], pos[5]] == 1  # bidirected in the ancestral ADMG
    assert pattern[pos[0], pos[2]] == 2  # bow, not a bidirected edge
    # hidden 2 and 5: no bidirected edge, but the noises of 2 and 5 still enter 0, 4 and 3, 4 (bows)
    ill2 = ConfoundingIllustration(paper_dag(), hidden=[2, 5])
    p2 = ill2.omega_pattern()
    pos2 = {v: k for k, v in enumerate(ill2.observed)}
    assert (p2 == 2).sum() == 4 and p2[pos2[0], pos2[4]] == 2 and p2[pos2[3], pos2[4]] == 2
    assert ill2.bidirected == []


def test_implicit_weight_pattern():
    root = ConfoundingIllustration(paper_dag(), hidden=[1])
    obs = root.observed
    # hidden root: W of the implicit model equals the observed block W_OO
    assert (root.implicit_weight_pattern() == root.adj[np.ix_(obs, obs)]).all()
    inter = ConfoundingIllustration(paper_dag(), hidden=[2, 5])
    w = inter.implicit_weight_pattern()
    pos = {v: k for k, v in enumerate(inter.observed)}
    for parent, child in [(1, 0), (1, 3), (1, 4)]:  # paths 1->2->0, 1->5->3, 1->2->4 are absorbed
        assert w[pos[child], pos[parent]] == 1
    assert w[pos[3], pos[1]] == 1 and inter.adj[3, 1] == 0  # not a direct edge of the DAG


def test_layout_and_outputs():
    adj = paper_dag()
    pos = layered_positions(adj)
    assert set(pos) == set(range(7))
    ill = ConfoundingIllustration(adj, hidden=[2, 5])
    assert ill.to_matplotlib() is not None
    tex = ill.to_tikz()
    assert tex.count("\\begin{tikzpicture}") == 3
    assert "W_{OU}" in tex and "\\node[hid]" in tex


def test_implicit_weight_pattern_marked_and_two_lines():
    import numpy as np
    from causalspyne.illustration import ConfoundingIllustration
    edges = [(1, 0), (1, 2), (1, 5), (2, 0), (2, 4), (5, 3), (5, 4), (6, 3), (6, 4), (0, 4), (3, 4)]
    adj = np.zeros((7, 7), dtype=int)
    for p, c in edges:
        adj[c, p] = 1
    ill = ConfoundingIllustration(adj, hidden=[2, 5])
    pat = ill.implicit_weight_pattern_marked()
    obs = ill.observed
    # 1 -> 3 and 1 -> 4 appear only through hidden paths; 1 -> 0 is also a sub-DAG edge
    assert pat[obs.index(3), obs.index(1)] == 2 and pat[obs.index(4), obs.index(1)] == 2
    assert pat[obs.index(0), obs.index(1)] == 1
    assert "observed sub-DAG" in ill.to_tikz(two_lines=True)


def test_subdag_panel_has_no_hidden_vertices():
    import numpy as np
    from causalspyne.illustration import ConfoundingIllustration
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = adj[2, 1] = 1  # 1 -> 0, 1 -> 2 ; hide vertex 1
    ill = ConfoundingIllustration(adj, hidden=[1])
    code = ill._tikz_graph(True, 1, 1, sub=True)
    assert "(n1)" not in code and "\\draw" not in code


def test_panels_tikz_separate_files():
    import numpy as np
    from causalspyne.illustration import ConfoundingIllustration
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = adj[2, 1] = 1
    panels = ConfoundingIllustration(adj, hidden=[1]).panels_tikz()
    assert set(panels) == {"dag", "admg", "explicit_W", "implicit_W", "implicit_Omega"}
    assert all(p.startswith("\\begin{tikzpicture}") and p.rstrip().endswith("\\end{tikzpicture}") for p in panels.values())
    assert "gray" in panels["dag"]  # edges at the hidden vertex are gray
