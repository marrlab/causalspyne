import numpy as np
from numpy.random import default_rng

from causalspyne.dag_gen_topo_order import RandTopoOrderDAG
from causalspyne.is_dag import is_dag


def test_topo_order_dag_is_dag():
    gen = RandTopoOrderDAG(default_rng(0))
    for num_nodes, degree in [(3, 2), (10, 3), (20, 4)]:
        mat = gen(num_nodes, degree)
        assert is_dag(mat), f"Not a DAG for num_nodes={num_nodes}, degree={degree}"


def test_topo_order_dag_no_self_loops():
    gen = RandTopoOrderDAG(default_rng(1))
    mat = gen(10, 3)
    assert np.trace(mat) == 0


def test_topo_order_dag_expected_edge_density():
    rng = default_rng(42)
    gen = RandTopoOrderDAG(rng)
    degree = 3
    num_nodes = 20
    n_trials = 50
    edge_counts = [gen(num_nodes, degree).sum() for _ in range(n_trials)]
    expected = degree * num_nodes / 2
    assert abs(np.mean(edge_counts) - expected) < expected * 0.3
