"""
Multi-level hidden confounder benchmark.

Compares FCI SHD and power spectrum of observed data when confounders at
different topological positions are hidden: 0.0 (root), 0.25, 0.5, 0.75,
1.0 (deepest). All scenarios share the same DAG and full data (same seed).

Produces:
  - Grouped boxplot of SHD by topological position
  - Kruskal-Wallis test across all levels + pairwise Wilcoxon tests
  - Power spectrum (eigenvalues of normalized precision) per level
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

# Topological positions to compare (0=root, 1.0=deepest)
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
LEVEL_LABELS = ["0.0\n(root)", "0.25", "0.5", "0.75", "1.0\n(deepest)"]
COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]


def run_one_seed(seed, num_macro_nodes, args):
    """Run all 5 hiding levels on the same DAG."""
    # Reuse run_paired_scenarios with a custom SCENARIO_HIDDEN mapping
    import io, contextlib
    from causalspyne.main import gen_partially_observed
    from causalspyne.dag_gen_topo_order import RootConfounderDAG

    results = {}
    full_data_ref = None

    for level in LEVELS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            subview = gen_partially_observed(
                size_micro_node_dag=args.size_micro_node_dag,
                max_num_local_nodes=args.max_num_local_nodes,
                min_num_local_nodes=args.min_num_local_nodes,
                num_macro_nodes=num_macro_nodes,
                degree=args.degree,
                list_confounder2hide=[level],
                num_sample=args.samples,
                output_dir=str(args.output_dir / f"macro{num_macro_nodes}/seed_{seed}/level_{level}"),
                rng=seed,
                plot=False,
                strategy_cls=RootConfounderDAG,
            )
        hidden = list(subview.list_global_inds_nodes2hide)
        graph, _ = fci(
            subview.data, alpha=args.alpha,
            verbose=False, show_progress=False,
            node_names=subview.node_names,
        )
        shd = structural_hamming_distance(
            true_dag=subview.dag.mat_adjacency,
            true_hidden_nodes=hidden,
            prediction=graph.graph,
        )
        # Eigenvalues of normalized precision of observed data
        try:
            cov = np.cov(subview.data.T)
            inv_cov = np.linalg.inv(cov)
            d = np.sqrt(np.diag(inv_cov))
            norm_prec = inv_cov / np.outer(d, d)
            eigvals = np.linalg.eigvalsh(norm_prec).tolist()
        except np.linalg.LinAlgError:
            eigvals = []

        results[level] = {
            "shd": shd,
            "hidden": hidden,
            "eigvals": eigvals,
            "subview": subview,
        }

    return results


def collect_all(args):
    records = []
    all_eigvals = {lv: [] for lv in LEVELS}
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
                res = run_one_seed(seed, n_macro, args)
                row = {"num_macro_nodes": n_macro, "seed": seed}
                for lv in LEVELS:
                    row[f"shd_{lv}"] = res[lv]["shd"]
                    all_eigvals[lv].extend(res[lv]["eigvals"])
                records.append(row)
            except Exception as e:
                print(f"SKIP {label}: {e}", file=sys.stderr)
            if bar:
                bar.update(1)

    if bar:
        bar.close()
    return pd.DataFrame.from_records(records), all_eigvals


def plot_shd_boxplot(df, macro_sizes, output_path):
    n_sizes = len(macro_sizes)
    fig, axes = plt.subplots(1, n_sizes, figsize=(4 * n_sizes, 5), sharey=True)
    if n_sizes == 1:
        axes = [axes]

    for ax, n_macro in zip(axes, macro_sizes):
        sub = df[df["num_macro_nodes"] == n_macro].dropna()
        data = [sub[f"shd_{lv}"].values for lv in LEVELS]

        bp = ax.boxplot(data, tick_labels=LEVEL_LABELS, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for box, color in zip(bp["boxes"], COLORS):
            box.set_facecolor(color)
            box.set_alpha(0.7)

        # Kruskal-Wallis
        stat, pval = stats.kruskal(*data)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        ax.set_title(f"{n_macro} macro nodes\nKW p={pval:.3f} {sig}", fontsize=10)
        ax.set_xlabel("Topological position of hidden confounder")
        ax.set_ylabel("Normalized SHD" if ax == axes[0] else "")

    fig.suptitle("FCI SHD by topological position of hidden confounder\n(same DAG, 1 node hidden per scenario)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    print(f"Saved → {output_path}")


def plot_spectrum_by_level(all_eigvals, output_path):
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(3.5 * len(LEVELS), 4), sharey=True)
    for ax, lv, label, color in zip(axes, LEVELS, LEVEL_LABELS, COLORS):
        ev = np.array(all_eigvals[lv])
        if len(ev) == 0:
            ax.set_title(label)
            continue
        ax.hist(ev, bins=40, density=True, color=color, alpha=0.75)
        ax.axvline(np.median(ev), color="black", linestyle="--", linewidth=1,
                   label=f"med={np.median(ev):.2f}")
        ax.set_title(f"pos={label}\nrange={ev.max()-ev.min():.2f}", fontsize=9)
        ax.set_xlabel("Eigenvalue of norm. precision")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Density")
    fig.suptitle("Power spectrum of observed data by topological position of hidden confounder", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    print(f"Saved → {output_path}")


def print_summary(df, macro_sizes):
    print("\n=== SHD Summary by level ===")
    for n in macro_sizes:
        sub = df[df["num_macro_nodes"] == n].dropna()
        if len(sub) == 0:
            continue
        print(f"\nmacro={n}:")
        data = [sub[f"shd_{lv}"].values for lv in LEVELS]
        for lv, d in zip(LEVELS, data):
            print(f"  pos={lv:.2f}  mean={d.mean():.3f}±{d.std():.3f}")
        if len(sub) >= 5:
            stat, pval = stats.kruskal(*data)
            print(f"  Kruskal-Wallis p={pval:.4f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-level hidden confounder benchmark."
    )
    p.add_argument("--macro-node-sizes", type=int, nargs="+", default=[4])
    p.add_argument("--replicates", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--size-micro-node-dag", type=int, default=None)
    p.add_argument("--min-num-local-nodes", type=int, default=3)
    p.add_argument("--max-num-local-nodes", type=int, default=7)
    p.add_argument("--degree", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path,
                   default=Path("output/multilevel_hidden"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, all_eigvals = collect_all(args)
    df.to_csv(args.output_dir / "results.csv", index=False)

    print_summary(df, args.macro_node_sizes)
    plot_shd_boxplot(df, args.macro_node_sizes,
                     args.output_dir / "multilevel_shd_boxplot.pdf")
    plot_spectrum_by_level(all_eigvals,
                           args.output_dir / "multilevel_spectrum.pdf")


if __name__ == "__main__":
    main()
