"""
Direct test: does spectral range of observed precision correlate with FCI SHD?

For each random explicit-DGP instance we compute:
  - spectral_range = lambda_max - lambda_min  of the normalized precision of observed data
  - FCI SHD against the true ADMG (root cluster hidden)

Then we test:
  - Spearman rank correlation (spectral_range, shd)
  - Kruskal-Wallis across spectral-range quartile bins

No rejection sampling needed: natural variation across graph sizes, degrees,
and seeds provides a wide range of spectral radii.

Usage:
    uv run python examples/compare_spectrum_radius_vs_shd.py [--output-dir ...]
"""
from __future__ import annotations

import argparse
import io
import contextlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from causallearn.search.ConstraintBased.FCI import fci

from causalspyne.implicit_gen_Sigma import cov_to_corr
from causalspyne.main import gen_partially_observed
from causalspyne.dag_gen_topo_order import RootConfounderDAG
from causalspyne.ancestral_shd import structural_hamming_distance

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def spectral_range_of_observed(data: np.ndarray) -> float:
    """Spectral range (lambda_max - lambda_min) of normalized precision of observed data."""
    try:
        cov = np.cov(data.T)
        prec = np.linalg.inv(cov)
        norm_prec = cov_to_corr(prec)          # normalized precision = partial-corr matrix (up to sign)
        eigvals = np.linalg.eigvalsh(norm_prec)
        return float(eigvals.max() - eigvals.min()), eigvals.tolist()
    except np.linalg.LinAlgError:
        return float("nan"), []


def run_one(seed, num_macro_nodes, degree, args) -> dict | None:
    """Generate one explicit instance, run FCI, return spectral range + SHD."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            subview = gen_partially_observed(
                size_micro_node_dag=None,
                max_num_local_nodes=args.max_num_local_nodes,
                min_num_local_nodes=args.min_num_local_nodes,
                num_macro_nodes=num_macro_nodes,
                degree=degree,
                list_confounder2hide=[0.0],   # always hide root cluster
                num_sample=args.samples,
                output_dir=str(
                    args.output_dir
                    / f"raw/macro{num_macro_nodes}/deg{degree}/seed{seed}"
                ),
                rng=seed,
                plot=False,
                strategy_cls=RootConfounderDAG,
            )
    except Exception as e:
        print(f"  gen error (macro={num_macro_nodes} deg={degree} seed={seed}): {e}",
              file=sys.stderr)
        return None

    srange, eigvals = spectral_range_of_observed(subview.data)
    if np.isnan(srange):
        return None

    hidden = list(subview.list_global_inds_nodes2hide)
    try:
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
    except Exception as e:
        print(f"  fci error (macro={num_macro_nodes} deg={degree} seed={seed}): {e}",
              file=sys.stderr)
        return None

    return {
        "seed": seed,
        "num_macro_nodes": num_macro_nodes,
        "degree": degree,
        "num_observed": subview.data.shape[1],
        "spectral_range": srange,
        "shd": shd,
    }


def collect(args):
    combos = [
        (seed, n_macro, deg)
        for n_macro in args.macro_node_sizes
        for deg in args.degrees
        for seed in range(args.seed_start, args.seed_start + args.seeds_per_combo)
    ]

    bar = tqdm(total=len(combos), unit="run") if tqdm else None
    records = []
    for seed, n_macro, deg in combos:
        label = f"macro={n_macro} deg={deg} seed={seed}"
        if bar:
            bar.set_postfix_str(label, refresh=True)
        else:
            print(label, file=sys.stderr, flush=True)
        rec = run_one(seed, n_macro, deg, args)
        if rec:
            records.append(rec)
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    return pd.DataFrame.from_records(records)


def plot_scatter(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: scatter with regression line
    ax = axes[0]
    x, y = df["spectral_range"].values, df["shd"].values
    ax.scatter(x, y, alpha=0.35, s=18, color="steelblue")
    m, b, r, p, _ = stats.linregress(x, y)
    xgrid = np.linspace(x.min(), x.max(), 200)
    ax.plot(xgrid, m * xgrid + b, color="firebrick", linewidth=1.5,
            label=f"OLS: slope={m:.3f}\nr={r:.3f}, p={p:.4f}")
    ax.set_xlabel("Spectral range of observed precision\n"
                  r"($\lambda_{\max} - \lambda_{\min}$ of normalized $\Sigma^{-1}$)")
    ax.set_ylabel("Normalized SHD (FCI)")
    ax.set_title("Spectral range vs. FCI SHD")
    ax.legend(fontsize=9)

    # Right: boxplot across quartile bins
    ax2 = axes[1]
    labels, bins = pd.qcut(df["spectral_range"], q=4, labels=False, retbins=True)
    groups = [df["shd"][labels == q].values for q in range(4)]
    bin_labels = [f"Q{i+1}\n[{bins[i]:.2f},{bins[i+1]:.2f})" for i in range(4)]
    bp = ax2.boxplot(groups, tick_labels=bin_labels, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2))
    colors = ["#4dac26", "#b8e186", "#f1b6da", "#d01c8b"]
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c); box.set_alpha(0.7)
    kw_stat, kw_p = stats.kruskal(*groups)
    ax2.set_xlabel("Spectral range quartile")
    ax2.set_ylabel("Normalized SHD (FCI)")
    ax2.set_title(f"SHD by spectral-range quartile\nKW p={kw_p:.4f}")

    fig.suptitle("FCI performance vs. spectral range of observed precision matrix\n"
                 "(explicit DGP: root cluster hidden)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


def print_summary(df):
    x, y = df["spectral_range"].values, df["shd"].values
    rho, sp = stats.spearmanr(x, y)
    r, pp = stats.pearsonr(x, y)
    print(f"\n=== Spectral range vs. SHD ({len(df)} instances) ===")
    print(f"  Spearman ρ = {rho:.4f}  p = {sp:.6f}")
    print(f"  Pearson  r = {r:.4f}  p = {pp:.6f}")
    print(f"  Spectral range: min={x.min():.3f}  max={x.max():.3f}  "
          f"mean={x.mean():.3f}  std={x.std():.3f}")
    print(f"  SHD:            min={y.min():.3f}  max={y.max():.3f}  "
          f"mean={y.mean():.3f}  std={y.std():.3f}")

    labels = pd.qcut(df["spectral_range"], q=4, labels=False)
    bins_ = pd.qcut(df["spectral_range"], q=4).cat.categories
    groups = [df["shd"][labels == q].values for q in range(4)]
    kw_stat, kw_p = stats.kruskal(*groups)
    print(f"\n  Kruskal-Wallis across spectral-range quartiles: p = {kw_p:.6f}")
    for q, grp, intv in zip(range(4), groups, bins_):
        print(f"    Q{q+1} range={intv}  n={len(grp)}  SHD mean={grp.mean():.3f}±{grp.std():.3f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Test if spectral range of observed precision correlates with FCI SHD."
    )
    p.add_argument("--macro-node-sizes", type=int, nargs="+", default=[3, 4])
    p.add_argument("--degrees", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    p.add_argument("--seeds-per-combo", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--min-num-local-nodes", type=int, default=3)
    p.add_argument("--max-num-local-nodes", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path, default=Path("output/spectrum_vs_shd"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(args.macro_node_sizes) * len(args.degrees) * args.seeds_per_combo} instances...")
    df = collect(args)
    df.to_csv(args.output_dir / "results.csv", index=False)
    print(f"Collected {len(df)} valid instances.")

    print_summary(df)
    plot_scatter(df, args.output_dir / "spectrum_range_vs_shd.pdf")


if __name__ == "__main__":
    main()
