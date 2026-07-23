"""
ADMG diversity comparison: two-level vs single-level explicit DGP.

Measures structural diversity of the ancestral ADMGs produced by each method:
  - Number of bidirected edges (confounding pairs visible to algorithms)
  - Number of directed edges in ADMG
  - Fraction of node pairs that are confounded

Hypothesis: the two-level block-hierarchical structure produces a wider
distribution of ADMG structures (more varied confounder patterns), while
single-level generation with a root confounder produces more homogeneous
graphs (always one cluster of confounded children).

Usage:
    uv run python examples/compare_admg_diversity.py
    uv run python examples/compare_admg_diversity.py --trials 200 --output-dir output/admg_diversity
"""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from causalspyne.main import gen_partially_observed
from causalspyne.dag_gen_topo_order import RootConfounderDAG


def admg_stats_one(seed: int, num_macro: int, min_micro: int, max_micro: int,
                   degree: float, list_confounder2hide: list) -> dict | None:
    """Generate one instance and return ADMG structural stats."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            subview = gen_partially_observed(
                size_micro_node_dag=None,
                min_num_local_nodes=min_micro,
                max_num_local_nodes=max_micro,
                num_macro_nodes=num_macro,
                degree=degree,
                list_confounder2hide=list_confounder2hide,
                num_sample=10,
                output_dir=str(Path("/tmp") / f"admg_{num_macro}m_{seed}"),
                rng=seed,
                plot=False,
                strategy_cls=RootConfounderDAG,
            )
    except Exception:
        return None

    # The ADMG adjacency can be inferred from the dag mat_adjacency + hidden nodes
    dag_adj = subview.dag.mat_adjacency   # full DAG adjacency (rows = parents)
    hidden = list(subview.list_global_inds_nodes2hide)
    n_total = dag_adj.shape[0]
    observed = [i for i in range(n_total) if i not in hidden]
    p_obs = len(observed)

    # Count bidirected edges: pairs (i,j) in observed that share a hidden common cause
    bidirected = 0
    for h in hidden:
        children = [i for i in observed if dag_adj[i, h] != 0]  # h -> i
        bidirected += len(children) * (len(children) - 1) // 2

    # Count directed edges among observed in ADMG
    # d_i -> d_j if i -> j path exists that does not go through hidden nodes
    # Simplified: just count direct observed->observed edges in original DAG
    directed = 0
    for i in observed:
        for j in observed:
            if dag_adj[j, i] != 0:  # i -> j
                directed += 1

    return {
        "seed": seed,
        "num_macro": num_macro,
        "n_obs": p_obs,
        "bidirected_edges": bidirected,
        "directed_edges": directed,
        "bidirected_fraction": bidirected / max(1, p_obs * (p_obs - 1) / 2),
    }


def collect(args, label: str, num_macro: int, min_micro: int, max_micro: int,
            hide: list) -> pd.DataFrame:
    records = []
    seed = 0
    while len(records) < args.trials:
        r = admg_stats_one(seed, num_macro, min_micro, max_micro, args.degree, hide)
        if r:
            r["method"] = label
            records.append(r)
        seed += 1
        if seed > args.trials * 5:
            break
    return pd.DataFrame.from_records(records)


def plot_and_save(df: pd.DataFrame, out_dir: Path) -> Path:
    methods = df["method"].unique()
    colours = {"Single-level (root)": "#a1d99b",
               "Two-level (root)": "#fdae6b",
               "Two-level (intermediate)": "#9ecae1"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col, ylabel in zip(
        axes,
        ["bidirected_edges", "directed_edges", "bidirected_fraction"],
        ["# Bidirected edges", "# Directed edges (obs→obs)", "Fraction confounded pairs"],
    ):
        data = [df[df["method"] == m][col].values for m in methods]
        bp = ax.boxplot(data, tick_labels=methods, patch_artist=True)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(colours.get(m, "#cccccc"))
            patch.set_alpha(0.8)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(methods, rotation=12, fontsize=8)

        # KW test
        kw_stat, kw_p = stats.kruskal(*data)
        ax.set_title(f"KW p={kw_p:.4f}", fontsize=9)

    fig.suptitle("ADMG structural diversity: single-level vs two-level DGP", fontsize=12)
    fig.tight_layout()
    out_pdf = out_dir / "admg_diversity.pdf"
    out_png = out_dir / "admg_diversity.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--degree", type=float, default=2.0)
    ap.add_argument("--output-dir", type=Path, default=Path("output/admg_diversity"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.trials} instances per configuration ...")
    dfs = []

    # Single-level: 1 macro node, several observed, 1 hidden root
    print("  Single-level (root hidden) ...")
    dfs.append(collect(args, "Single-level (root)", num_macro=1,
                       min_micro=6, max_micro=6, hide=[0.0]))

    # Two-level: 3 macro nodes, 2-4 micro each, root hidden
    print("  Two-level (root hidden) ...")
    dfs.append(collect(args, "Two-level (root)", num_macro=3,
                       min_micro=2, max_micro=4, hide=[0.0]))

    # Two-level: intermediate hidden
    print("  Two-level (intermediate hidden) ...")
    dfs.append(collect(args, "Two-level (intermediate)", num_macro=3,
                       min_micro=2, max_micro=4, hide=[1.0]))

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(args.output_dir / "results.csv", index=False)

    print("\n=== Summary ===")
    for m in df["method"].unique():
        sub = df[df["method"] == m]
        print(f"  {m}: bidir={sub['bidirected_edges'].mean():.2f}±{sub['bidirected_edges'].std():.2f}  "
              f"dir={sub['directed_edges'].mean():.2f}±{sub['directed_edges'].std():.2f}")

    out_png = plot_and_save(df, args.output_dir)
    return out_png


if __name__ == "__main__":
    main()
