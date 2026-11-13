"""
test if shd runs
"""

import numpy as np
from causalspyne.ancestral_shd import structural_hamming_distance


def test_shd():
    """
    test shd runs
    """
    target_graph = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    predicted_graph = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    shd = structural_hamming_distance(target_graph, [], predicted_graph)
    print(f"Structural Hamming Distance: {shd}")


def test_shd_with_hidden_node():
    target_graph = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    predicted_graph = np.array([[0, 1], [1, 0]])

    shd = structural_hamming_distance(target_graph, [0], predicted_graph)

    assert np.isfinite(shd)
    assert 0 <= shd <= 1


def _perfect_prediction(true_dag, hidden):
    """true PAG built independently, in observed (ascending index) order"""
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.DAG2PAG import dag2pag

    n = len(true_dag)
    nodes = [GraphNode(f"X{i}") for i in range(n)]
    dag = Dag(nodes)
    for ch, pa in np.argwhere(true_dag):
        dag.add_directed_edge(nodes[pa], nodes[ch])
    pag = dag2pag(dag, [nodes[i] for i in hidden])
    observed = [i for i in range(n) if i not in hidden]
    names = [f"X{i}" for i in observed]
    perm = [names.index(v.get_name()) for v in pag.get_nodes()]
    mat = np.zeros_like(pag.graph)
    mat[np.ix_(perm, perm)] = pag.graph
    return mat


def test_shd_perfect_prediction_is_zero():
    """regression: dag2pag returns nodes in arbitrary order; SHD of the true
    PAG against itself (in data column order) must be 0"""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        true_dag = np.tril(rng.random((10, 10)) < 0.3, -1).astype(int)
        for hidden in ([], [0], [0, 3]):
            pred = _perfect_prediction(true_dag, hidden)
            assert structural_hamming_distance(true_dag, hidden, pred) == 0
