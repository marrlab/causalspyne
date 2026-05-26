"""
Grouped paired boxplot: root vs intermediate hidden confounder across graph sizes.

For each graph size (num_macro_nodes), runs the fair single-hidden benchmark
(same DAG, 1 node hidden per scenario). Produces:
  - Grouped boxplot with paired lines connecting same-seed points
  - Wilcoxon signed-rank test (paired) per group
  - Summary CSV
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
from scipy import stats
from causallearn.search.ConstraintBased.FCI import fci

from causalspyne.benchmark_fci import run_paired_scenarios, SCENARIO_HIDDEN
from causalspyne.ancestral_shd import structural_hamming_distance

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

SCENARIOS = {
    "root":         {"label": "Root hidden"},
    "intermediate": {"label": "Intermediate hidden"},
}


def run_one_seed(seed, num_macro_nodes, args):
    """Run both scenarios on the same DAG using core benchmark logic."""
    paired = run_paired_scenarios(
        seed=seed,
        num_macro_nodes=num_macro_nodes,
        size_micro_node_dag=args.size_micro_node_dag,
        max_num_local_nodes=args.max_num_local_nodes,
        degree=args.degree,
        num_sample=args.samples,
        output_dir=str(args.output_dir / f"macro{num_macro_nodes}"),
    )
    results = {}
    for scenario_name, data in paired.items():
        subview = data["subview"]
        graph, _ = fci(
            subview.data, alpha=args.alpha,
            verbose=False, show_progress=False,
            node_names=subview.node_names,
        )
        results[scenario_name] = structural_hamming_distance(
            true_dag=subview.dag.mat_adjacency,
            true_hidden_nodes=data["hidden"],
            prediction=graph.graph,
        )
    return results


def collect_all(args):
    records = []
    total = len(args.macro_node_sizes) * args.replicates
    bar = tqdm(total=total, unit="run") if tqdm else None

    for n_macro in args.macro_node_sizes:
        seeds = range(args.seed_start, args.seed_start + args.replicates)
        for seed in seeds:
            label = f"macro={n_macro} seed={seed}"
            if bar:
                bar.set_postfix_str(label, refresh=True)
            else:
                print(label, file=sys.stderr, flush=True)

            try:
                shds = run_one_seed(seed, n_macro, args)
                records.append({
                    "num_macro_nodes": n_macro,
                    "seed": seed,
                    "root_shd": shds["root"],
                    "intermediate_shd": shds["intermediate"],
                })
            except Exception as e:
                print(f"SKIP {label}: {e}", file=sys.stderr)

            if bar:
                bar.update(1)

    if bar:
        bar.close()
    return pd.DataFrame.from_records(records)


def plot_grouped_paired(df, macro_sizes, output_path):
    n_sizes = len(macro_sizes)
    fig, axes = plt.subplots(1, n_sizes, figsize=(4 * n_sizes, 5), sharey=True)
    if n_sizes == 1:
        axes = [axes]

    for ax, n_macro in zip(axes, macro_sizes):
        sub = df[df["num_macro_nodes"] == n_macro].dropna()
        root_vals = sub["root_shd"].values
        inter_vals = sub["intermediate_shd"].values

        bp = ax.boxplot(
            [root_vals, inter_vals],
            tick_labels=["Root\nhidden", "Intermediate\nhidden"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#4C9BE8")
        bp["boxes"][1].set_facecolor("#F4A261")

        # paired lines
        x_root = np.random.default_rng(42).normal(1, 0.04, len(root_vals))
        x_inter = np.random.default_rng(43).normal(2, 0.04, len(inter_vals))
        for xr, xi, yr, yi in zip(x_root, x_inter, root_vals, inter_vals):
            ax.plot([xr, xi], [yr, yi], color="grey", alpha=0.35,
                    linewidth=0.8, zorder=2)
        ax.scatter(x_root, root_vals, color="#1A5EA8", s=18, zorder=3, alpha=0.8)
        ax.scatter(x_inter, inter_vals, color="#C1440E", s=18, zorder=3, alpha=0.8)

        # Wilcoxon signed-rank test
        if len(root_vals) >= 5:
            stat, pval = stats.wilcoxon(root_vals, inter_vals)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
            ax.set_title(f"{n_macro} macro nodes\np={pval:.3f} {sig}", fontsize=10)
        else:
            ax.set_title(f"{n_macro} macro nodes", fontsize=10)

        ax.set_ylabel("Normalized SHD" if ax == axes[0] else "")

    fig.suptitle("FCI: root vs intermediate hidden confounder\n(same DAG, 1 node hidden, paired)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    print(f"Saved → {output_path}")


def print_summary(df, macro_sizes):
    print("\n=== Summary ===")
    for n in macro_sizes:
        sub = df[df["num_macro_nodes"] == n].dropna()
        if len(sub) == 0:
            continue
        r, i = sub["root_shd"].values, sub["intermediate_shd"].values
        pval = stats.wilcoxon(r, i).pvalue if len(r) >= 5 else float("nan")
        print(f"macro={n}  root={r.mean():.3f}±{r.std():.3f}  "
              f"inter={i.mean():.3f}±{i.std():.3f}  "
              f"Wilcoxon p={pval:.4f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Grouped paired boxplot: root vs intermediate across graph sizes."
    )
    p.add_argument("--macro-node-sizes", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--replicates", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--size-micro-node-dag", type=int, default=None,
                   help="Fixed micro DAG size per macro node; None = random in [2, max-num-local-nodes]")
    p.add_argument("--max-num-local-nodes", type=int, default=7,
                   help="Upper bound on random micro DAG size when size-micro-node-dag is None")
    p.add_argument("--degree", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path,
                   default=Path("output/root_vs_intermediate_grouped"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_all(args)
    df.to_csv(args.output_dir / "results.csv", index=False)

    print_summary(df, args.macro_node_sizes)
    plot_grouped_paired(df, args.macro_node_sizes,
                        args.output_dir / "grouped_paired_boxplot.pdf")


if __name__ == "__main__":
    main()
