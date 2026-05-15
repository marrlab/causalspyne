"""
Compare FCI performance when hiding root-like vs intermediate confounders.

This script mirrors the Fig. 4/Fig. 5 comparison in the manuscript:

* standard/root: hide the first topologically sorted confounder
* intermediate: hide later topologically sorted confounders

It writes a CSV summary and a boxplot of normalized SHD.
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
from causalspyne.ancestral_shd import structural_hamming_distance

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_SCENARIOS = {
    "root": {
        "label": "Root / standard",
        "hidden": [0],
    },
    "intermediate": {
        "label": "Intermediate",
        "hidden": [0.7, 0.9],
    },
}


def _parse_hidden_positions(raw_values):
    hidden_positions = []
    for raw_value in raw_values:
        for raw_token in raw_value.split(","):
            token = raw_token.strip()
            if not token:
                raise argparse.ArgumentTypeError(
                    "hidden positions must not contain empty values"
                )
            try:
                if "." in token or "e" in token.lower():
                    hidden_positions.append(float(token))
                else:
                    hidden_positions.append(int(token))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"could not parse hidden position {token!r}"
                ) from exc
    return hidden_positions


def _build_scenarios(args):
    return {
        "root": {
            "label": DEFAULT_SCENARIOS["root"]["label"],
            "hidden": args.root_hidden,
        },
        "intermediate": {
            "label": DEFAULT_SCENARIOS["intermediate"]["label"],
            "hidden": args.intermediate_hidden,
        },
    }


def _node_summary(dag, hidden_global_indices):
    binary_adj = (dag.mat_adjacency != 0).astype(int)
    summaries = []
    for ind in hidden_global_indices:
        parents = np.flatnonzero(binary_adj[ind, :]).tolist()
        children = np.flatnonzero(binary_adj[:, ind]).tolist()
        summaries.append(
            {
                "index": int(ind),
                "name": dag.list_node_names[ind],
                "num_parents": len(parents),
                "num_children": len(children),
                "is_root": len(parents) == 0,
            }
        )
    return summaries


def _dag_signature(dag):
    return {
        "node_names": tuple(dag.list_node_names),
        "mat_adjacency": np.asarray(dag.mat_adjacency).copy(),
    }


def _assert_same_generated_dag(reference, candidate, seed, scenario_name):
    if reference["node_names"] != candidate["node_names"]:
        raise RuntimeError(
            "Generated DAG node names changed across scenarios for "
            f"seed {seed} before scenario {scenario_name!r}."
        )

    ref_adj = reference["mat_adjacency"]
    candidate_adj = candidate["mat_adjacency"]
    if ref_adj.shape == candidate_adj.shape:
        max_abs_diff = np.max(np.abs(ref_adj - candidate_adj))
    else:
        max_abs_diff = "n/a"

    if ref_adj.shape != candidate_adj.shape or not np.array_equal(
        ref_adj, candidate_adj
    ):
        raise RuntimeError(
            "Generated DAG changed across scenarios for "
            f"seed {seed} before scenario {scenario_name!r}; "
            "the comparison would not isolate hidden-node position. "
            f"reference_shape={ref_adj.shape}, "
            f"candidate_shape={candidate_adj.shape}, "
            f"max_abs_diff={max_abs_diff}"
        )


def run_one(args, scenario_name, scenario, seed, reference_dag=None):
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
        rng=seed,
        graphviz=args.graphviz,
        plot=args.plot_examples,
    )
    dag_signature = _dag_signature(subview.dag)
    if reference_dag is not None:
        _assert_same_generated_dag(reference_dag, dag_signature, seed, scenario_name)

    graph, _ = fci(
        subview.data,
        alpha=args.alpha,
        verbose=False,
        show_progress=args.fci_progress and not args.no_progress,
        node_names=subview.node_names,
    )

    hidden_global_indices = subview.list_global_inds_nodes2hide
    shd = structural_hamming_distance(
        true_dag=subview.dag.mat_adjacency,
        true_hidden_nodes=hidden_global_indices,
        prediction=graph.graph,
    )
    hidden_nodes = _node_summary(subview.dag, hidden_global_indices)

    record = {
        "scenario": scenario_name,
        "scenario_label": scenario["label"],
        "seed": seed,
        "normalized_shd": shd,
        "num_observed": subview.data.shape[1],
        "num_hidden": len(hidden_global_indices),
        "hidden_global_indices": ";".join(str(node["index"]) for node in hidden_nodes),
        "hidden_node_names": ";".join(node["name"] for node in hidden_nodes),
        "hidden_parent_counts": ";".join(
            str(node["num_parents"]) for node in hidden_nodes
        ),
        "hidden_child_counts": ";".join(
            str(node["num_children"]) for node in hidden_nodes
        ),
        "all_hidden_are_roots": all(node["is_root"] for node in hidden_nodes),
    }
    return record, dag_signature


def _progress_bar(total_runs, enabled):
    if not enabled or tqdm is None:
        return None
    return tqdm(total=total_runs, desc="FCI runs", unit="run")


def _report_run_start(progress_bar, enabled, run_index, total_runs, seed, scenario_name):
    if not enabled:
        return
    if progress_bar is not None:
        progress_bar.set_postfix(seed=seed, scenario=scenario_name, refresh=True)
        return
    print(
        f"[{run_index}/{total_runs}] seed={seed} scenario={scenario_name}",
        file=sys.stderr,
        flush=True,
    )


def plot_results(df, output_path, scenarios):
    labels = [scenarios[name]["label"] for name in scenarios]
    values = [
        df.loc[df["scenario"] == name, "normalized_shd"].to_numpy()
        for name in scenarios
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, tick_labels=labels)
    ax.scatter(
        np.concatenate(
            [np.full(len(group), idx + 1) for idx, group in enumerate(values)]
        ),
        np.concatenate(values),
        color="black",
        s=18,
        alpha=0.7,
        zorder=3,
    )
    ax.set_ylabel("Normalized SHD")
    ax.set_xlabel("Hidden confounder setting")
    ax.set_title("FCI sensitivity to hidden confounder position")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare FCI on data generated by hiding root-like and "
            "intermediate confounders."
        )
    )
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--size-micro-node-dag", type=int, default=3)
    parser.add_argument(
        "--random-micro-nodes",
        action="store_true",
        help=(
            "Use a random number of micro nodes per macro node instead of "
            "--size-micro-node-dag."
        ),
    )
    parser.add_argument("--max-num-local-nodes", type=int, default=4)
    parser.add_argument("--num-macro-nodes", type=int, default=3)
    parser.add_argument("--degree", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--root-hidden",
        nargs="+",
        default=[str(value) for value in DEFAULT_SCENARIOS["root"]["hidden"]],
        metavar="POSITION",
        help=(
            "Topologically sorted confounder positions to hide in the root "
            "scenario. Integers are indexes; decimals are quantiles. "
            "Accepts space- or comma-separated values. Default: 0."
        ),
    )
    parser.add_argument(
        "--intermediate-hidden",
        nargs="+",
        default=[
            str(value) for value in DEFAULT_SCENARIOS["intermediate"]["hidden"]
        ],
        metavar="POSITION",
        help=(
            "Topologically sorted confounder positions to hide in the "
            "intermediate scenario. Integers are indexes; decimals are "
            "quantiles. Accepts space- or comma-separated values. "
            "Default: 0.7 0.9."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/fci_confounder_position"),
    )
    parser.add_argument(
        "--plot-examples",
        action="store_true",
        help="Also write DAG/ancestral/subDAG comparison figures for each run.",
    )
    parser.add_argument(
        "--graphviz",
        action="store_true",
        help="Use graphviz layouts for generated DAG comparison figures.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable experiment progress reporting and causal-learn progress output.",
    )
    parser.add_argument(
        "--fci-progress",
        action="store_true",
        help=(
            "Also show causal-learn's internal per-depth/node progress messages "
            "(for example, 'Depth=0, working on node 8')."
        ),
    )
    args = parser.parse_args()
    args.root_hidden = _parse_hidden_positions(args.root_hidden)
    args.intermediate_hidden = _parse_hidden_positions(args.intermediate_hidden)
    if args.random_micro_nodes:
        args.size_micro_node_dag = None
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = _build_scenarios(args)

    records = []
    seeds = range(args.seed_start, args.seed_start + args.replicates)
    progress_enabled = not args.no_progress
    total_runs = args.replicates * len(scenarios)
    run_index = 0
    progress_bar = _progress_bar(total_runs, progress_enabled)
    try:
        for seed in seeds:
            reference_dag = None
            for scenario_name, scenario in scenarios.items():
                run_index += 1
                _report_run_start(
                    progress_bar,
                    progress_enabled,
                    run_index,
                    total_runs,
                    seed,
                    scenario_name,
                )
                record, dag_signature = run_one(
                    args, scenario_name, scenario, seed, reference_dag=reference_dag
                )
                if reference_dag is None:
                    reference_dag = dag_signature
                records.append(record)
                if progress_bar is not None:
                    progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    df = pd.DataFrame.from_records(records)
    csv_path = args.output_dir / "fci_confounder_position_results.csv"
    pdf_path = args.output_dir / "fci_confounder_position_boxplot.pdf"
    df.to_csv(csv_path, index=False)
    plot_results(df, pdf_path, scenarios)

    summary = df.groupby("scenario_label")["normalized_shd"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    print(summary)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
