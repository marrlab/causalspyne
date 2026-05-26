"""
Fair FCI benchmark: hide exactly 1 variable — root confounder vs intermediate.

Both scenarios share the SAME DAG (generated with RootConfounderDAG backbone
so the root is guaranteed to be a confounder). The only difference is WHICH
single confounder is hidden:

  - root:         hide the topologically first confounder (index 0)
  - intermediate: hide the topologically last confounder (index -1 / 1.0)

Same DAG, same number of observed variables, different hidden position.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci

from causalspyne.main import gen_partially_observed
from causalspyne.dag_gen_topo_order import RootConfounderDAG
from causalspyne.ancestral_shd import structural_hamming_distance

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

SCENARIOS = {
    "root": {
        "label": "Root confounder hidden",
        "hidden": [0],       # topologically first confounder
    },
    "intermediate": {
        "label": "Intermediate confounder hidden",
        "hidden": [1.0],     # topologically last confounder
    },
}


def _node_summary(dag, hidden_global_indices):
    binary_adj = (dag.mat_adjacency != 0).astype(int)
    summaries = []
    for ind in hidden_global_indices:
        parents = np.flatnonzero(binary_adj[ind, :]).tolist()
        children = np.flatnonzero(binary_adj[:, ind]).tolist()
        summaries.append({
            "index": int(ind),
            "num_parents": len(parents),
            "num_children": len(children),
            "is_root": len(parents) == 0,
        })
    return summaries


def run_seed(args, seed):
    """Run both scenarios on the same DAG for a given seed."""
    records = []
    reference_adj = None

    for scenario_name, scenario in SCENARIOS.items():
        run_dir = args.output_dir / scenario_name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        subview = gen_partially_observed(
            size_micro_node_dag=args.size_micro_node_dag,
            max_num_local_nodes=args.max_num_local_nodes,
            num_macro_nodes=args.num_macro_nodes,
            degree=args.degree,
            list_confounder2hide=scenario["hidden"],
            num_sample=args.samples,
            output_dir=run_dir,
            rng=seed,                      # same seed → same DAG skeleton
            plot=False,
            strategy_cls=RootConfounderDAG,
        )

        # verify same DAG across scenarios
        current_adj = subview.dag.mat_adjacency
        if reference_adj is None:
            reference_adj = current_adj.copy()
        elif not np.array_equal(reference_adj, current_adj):
            print(f"WARNING: DAG differs across scenarios for seed {seed}",
                  file=sys.stderr)

        graph, _ = fci(
            subview.data,
            alpha=args.alpha,
            verbose=False,
            show_progress=False,
            node_names=subview.node_names,
        )

        hidden_global_indices = subview.list_global_inds_nodes2hide
        shd = structural_hamming_distance(
            true_dag=subview.dag.mat_adjacency,
            true_hidden_nodes=hidden_global_indices,
            prediction=graph.graph,
        )
        node_info = _node_summary(subview.dag, hidden_global_indices)[0]
        records.append({
            "scenario": scenario_name,
            "scenario_label": scenario["label"],
            "seed": seed,
            "normalized_shd": shd,
            "num_observed": subview.data.shape[1],
            "hidden_global_ind": node_info["index"],
            "hidden_num_parents": node_info["num_parents"],
            "hidden_num_children": node_info["num_children"],
            "hidden_is_root": node_info["is_root"],
        })

    return records


def plot_results(df, output_path):
    scenario_names = list(SCENARIOS.keys())
    labels = [SCENARIOS[s]["label"] for s in scenario_names]
    values = [df.loc[df["scenario"] == s, "normalized_shd"].to_numpy()
              for s in scenario_names]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, tick_labels=labels)
    ax.scatter(
        np.concatenate([np.full(len(g), i + 1) for i, g in enumerate(values)]),
        np.concatenate(values),
        color="black", s=18, alpha=0.7, zorder=3,
    )
    ax.set_ylabel("Normalized SHD")
    ax.set_title("FCI: root vs intermediate hidden (1 node, same DAG)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fair FCI comparison: hide 1 node — root vs intermediate, same DAG."
    )
    p.add_argument("--replicates", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--size-micro-node-dag", type=int, default=3)
    p.add_argument("--max-num-local-nodes", type=int, default=4)
    p.add_argument("--num-macro-nodes", type=int, default=4)
    p.add_argument("--degree", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path,
                   default=Path("output/root_vs_intermediate_single"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seed_start, args.seed_start + args.replicates)

    records = []
    bar = tqdm(total=args.replicates, desc="seeds", unit="seed") if tqdm else None
    try:
        for seed in seeds:
            if bar:
                bar.set_postfix(seed=seed, refresh=True)
            else:
                print(f"seed={seed}", file=sys.stderr, flush=True)
            records.extend(run_seed(args, seed))
            if bar:
                bar.update(1)
    finally:
        if bar:
            bar.close()

    df = pd.DataFrame.from_records(records)
    csv_path = args.output_dir / "results.csv"
    pdf_path = args.output_dir / "boxplot.pdf"
    df.to_csv(csv_path, index=False)
    plot_results(df, pdf_path)

    summary = df.groupby("scenario_label")["normalized_shd"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    print(summary.to_string())
    print(f"\nWrote {csv_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
