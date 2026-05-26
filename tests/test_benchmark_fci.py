"""
Tests for the paired FCI benchmark invariant:
  same seed → same DAG, same data; only hidden columns differ.
"""

import numpy as np
import pytest
from causalspyne.benchmark_fci import run_paired_scenarios


@pytest.fixture(scope="module")
def paired_results():
    return run_paired_scenarios(
        seed=7,
        num_macro_nodes=3,
        size_micro_node_dag=3,
        num_sample=50,
        output_dir="/tmp/test_benchmark_fci",
    )


def test_same_dag_adjacency(paired_results):
    """Both scenarios must produce the exact same DAG structure."""
    root_adj = paired_results["root"]["adj"]
    inter_adj = paired_results["intermediate"]["adj"]
    assert root_adj.shape == inter_adj.shape, "DAG sizes differ"
    assert np.array_equal(root_adj, inter_adj), (
        "DAG adjacency matrices differ between root and intermediate scenario "
        "for the same seed — the benchmark is not comparing the same graph."
    )


def test_same_full_data(paired_results):
    """Both scenarios must be generated from the same data matrix."""
    root_data = paired_results["root"]["full_data"]
    inter_data = paired_results["intermediate"]["full_data"]
    assert root_data.shape == inter_data.shape, "Full data shapes differ"
    assert np.allclose(root_data, inter_data), (
        "Full data arrays differ between scenarios for the same seed — "
        "the benchmark is not hiding columns of the same dataset."
    )


def test_different_hidden_columns(paired_results):
    """The two scenarios must hide different columns."""
    root_hidden = set(paired_results["root"]["hidden"])
    inter_hidden = set(paired_results["intermediate"]["hidden"])
    assert root_hidden != inter_hidden, (
        "Root and intermediate scenarios hide the same columns — "
        "there is no contrast between the two conditions."
    )


def test_hidden_column_count(paired_results):
    """Each scenario hides exactly one variable."""
    assert len(paired_results["root"]["hidden"]) == 1
    assert len(paired_results["intermediate"]["hidden"]) == 1


def test_observed_data_differs(paired_results):
    """Observed (post-hide) data must differ between scenarios."""
    root_obs = paired_results["root"]["subview"].data
    inter_obs = paired_results["intermediate"]["subview"].data
    # Same number of observed columns (1 node hidden in each)
    assert root_obs.shape == inter_obs.shape
    # But the actual values differ because different columns were removed
    assert not np.allclose(root_obs, inter_obs), (
        "Observed data is identical across scenarios — "
        "hiding different columns should produce different observed datasets."
    )
