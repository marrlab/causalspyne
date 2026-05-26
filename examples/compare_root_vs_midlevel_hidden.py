"""
Compare FCI performance: hiding the root macro cluster vs a mid-level confounder.

Scenario A (root_hidden):
  Uses RootConfounderDAG — the macro backbone root is guaranteed to confound
  ≥2 other macro nodes. All micro nodes of that root cluster are hidden.

Scenario B (midlevel_hidden):
  Standard gen_partially_observed — hides a single confounder at the 50th
  percentile of the topologically sorted confounders.

The two scenarios use independent DAGs (different generators), so SHD
distributions are compared statistically rather than per-seed.
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

from causalspyne import gen_partially_observed
from causalspyne.main import gen_root_confounder_hidden
from causalspyne.ancestral_shd import structural_hamming_distance

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _node_summary(dag, hidden_global_indices):
    binary_adj = (dag.mat_adjacency != 0).astype(int)
    summaries = []
    for ind in hidden_global_indices:
        parents = np.flatnonzero(binary_adj[ind, :]).tolist()
        children = np.flatnonzero(binary_adj[:, ind]).tolist()
        summaries.append({
            "index": int(ind),
            "name": dag.list_node_names[ind],
            "num_parents": len(parents),
            "num_children": len(children),
            "is_root": len(parents) == 0,
        })
    return summaries


def run_root_hidden(args, seed):
    run_dir = args.output_dir / "root_hidden" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    subview = gen_root_confounder_hidden(
        size_micro_node_dag=args.size_micro_node_dag,
        max_num_local_nodes=args.max_num_local_nodes,
        num_macro_nodes=args.num_macro_nodes,
        degree=args.degree,
        num_sample=args.samples,
        output_dir=run_dir,
        rng=seed,
        plot=False,
    )

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
    hidden_nodes = _node_summary(subview.dag, hidden_global_indices)
    return {
        "scenario": "root_hidden",
        "scenario_label": "Root macro cluster hidden",
        "seed": seed,
        "normalized_shd": shd,
        "num_observed": subview.data.shape[1],
        "num_hidden": len(hidden_global_indices),
        "all_hidden_are_roots": all(n["is_root"] for n in hidden_nodes),
        "hidden_child_counts": ";".join(str(n["num_children"]) for n in hidden_nodes),
    }


def run_midlevel_hidden(args, seed):
    run_dir = args.output_dir / "midlevel_hidden" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    subview = gen_partially_observed(
        size_micro_node_dag=args.size_micro_node_dag,
        max_num_local_nodes=args.max_num_local_nodes,
        num_macro_nodes=args.num_macro_nodes,
        degree=args.degree,
        list_confounder2hide=[0.5],
        num_sample=args.samples,
        output_dir=run_dir,
        rng=seed,
        plot=False,
    )

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
    hidden_nodes = _node_summary(subview.dag, hidden_global_indices)
    return {
        "scenario": "midlevel_hidden",
        "scenario_label": "Mid-level confounder hidden",
        "seed": seed,
        "normalized_shd": shd,
        "num_observed": subview.data.shape[1],
        "num_hidden": len(hidden_global_indices),
        "all_hidden_are_roots": all(n["is_root"] for n in hidden_nodes),
        "hidden_child_counts": ";".join(str(n["num_children"]) for n in hidden_nodes),
    }


def plot_results(df, output_path):
    scenarios = ["root_hidden", "midlevel_hidden"]
    labels = ["Root macro cluster\nhidden", "Mid-level confounder\nhidden"]
    values = [df.loc[df["scenario"] == s, "normalized_shd"].to_numpy() for s in scenarios]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, tick_labels=labels)
    ax.scatter(
        np.concatenate([np.full(len(g), i + 1) for i, g in enumerate(values)]),
        np.concatenate(values),
        color="black", s=18, alpha=0.7, zorder=3,
    )
    ax.set_ylabel("Normalized SHD")
    ax.set_title("FCI difficulty: root vs mid-level hidden confounder")
    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FCI SHD: root macro cluster hidden vs mid-level confounder hidden."
    )
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--size-micro-node-dag", type=int, default=3)
    parser.add_argument("--max-num-local-nodes", type=int, default=4)
    parser.add_argument("--num-macro-nodes", type=int, default=4)
    parser.add_argument("--degree", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("output/root_vs_midlevel"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seed_start, args.seed_start + args.replicates)
    total_runs = args.replicates * 2

    records = []
    bar = tqdm(total=total_runs, desc="FCI runs", unit="run") if tqdm else None
    try:
        for seed in seeds:
            for runner, label in [(run_root_hidden, "root_hidden"),
                                  (run_midlevel_hidden, "midlevel_hidden")]:
                if bar:
                    bar.set_postfix(seed=seed, scenario=label, refresh=True)
                else:
                    print(f"seed={seed} scenario={label}", file=sys.stderr, flush=True)
                records.append(runner(args, seed))
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
