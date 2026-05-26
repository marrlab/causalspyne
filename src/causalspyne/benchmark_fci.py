"""
Core logic for the paired FCI benchmark (root vs intermediate hidden).

Factored out of examples/ so it can be unit-tested and reused.
The key invariant: both scenarios are generated from the same integer seed,
so they share the same DAG adjacency matrix and the same full data array;
only the hidden column indices differ.
"""

from __future__ import annotations

import numpy as np

from causalspyne.main import gen_partially_observed
from causalspyne.dag_gen_topo_order import RootConfounderDAG


SCENARIO_HIDDEN = {
    "root":         [0],    # topologically first confounder
    "intermediate": [1.0],  # topologically last confounder
}


def run_paired_scenarios(
    seed: int,
    num_macro_nodes: int = 4,
    size_micro_node_dag=None,
    max_num_local_nodes: int = 7,
    min_num_local_nodes: int = 3,
    degree: float = 2.0,
    num_sample: int = 200,
    output_dir: str = "/tmp/benchmark_fci",
    strategy_cls=None,
) -> dict:
    """
    Run both scenarios (root hidden, intermediate hidden) on the same DAG.

    Returns a dict with keys 'root' and 'intermediate', each containing:
      - 'subview':  the DAGView object (observed data + metadata)
      - 'full_data': np.ndarray of shape (num_sample, num_nodes_total)
                     — the data BEFORE any columns are hidden
      - 'adj':      binary adjacency matrix of the ground-truth DAG
      - 'hidden':   list of global node indices that were hidden

    Invariant (tested in tests/test_benchmark_fci.py):
      results['root']['adj']       == results['intermediate']['adj']
      results['root']['full_data'] == results['intermediate']['full_data']
      results['root']['hidden']    != results['intermediate']['hidden']
    """
    if strategy_cls is None:
        strategy_cls = RootConfounderDAG

    results = {}
    for scenario_name, hidden_spec in SCENARIO_HIDDEN.items():
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            subview = gen_partially_observed(
                size_micro_node_dag=size_micro_node_dag,
                max_num_local_nodes=max_num_local_nodes,
                min_num_local_nodes=min_num_local_nodes,
                num_macro_nodes=num_macro_nodes,
                degree=degree,
                list_confounder2hide=hidden_spec,
                num_sample=num_sample,
                output_dir=f"{output_dir}/{scenario_name}/seed_{seed}",
                rng=seed,          # integer → fresh RNG → reproducible
                plot=False,
                strategy_cls=strategy_cls,
            )

        results[scenario_name] = {
            "subview":   subview,
            "full_data": subview._data_arr,  # set by DAGView.run(), pre-hide
            "adj":       (subview.dag.mat_adjacency != 0).astype(int),
            "hidden":    list(subview.list_global_inds_nodes2hide),
        }

    return results
