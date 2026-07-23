"""
Three-way power-spectrum comparison:
  A. Implicit DGP   — diagonal-dominant Ω  → restricted eigenvalue range
  B. Single-level explicit — random DAG, 1 hidden root node → wider range
  C. Two-level explicit   — block-hierarchical (causalspyne) → widest range?

Hypothesis: two-level block structure introduces more spectral diversity because
the macro/micro hierarchy creates a richer covariance structure with multiple
confounder clusters at varied topological positions.

Usage:
    uv run python examples/compare_spectrum_twolevel_vs_singlelevel.py
    uv run python examples/compare_spectrum_twolevel_vs_singlelevel.py \\
        --trials 500 --output-dir output/spectrum_3way
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
from scipy import stats

from causalspyne.implicit_gen_Sigma import gen_sigma_y, cov_to_corr
from causalspyne.main import gen_partially_observed
from causalspyne.dag_gen_topo_order import RootConfounderDAG


# ── Implicit ────────────────────────────────────────────────────────────────

def implicit_eigenvalues_one() -> list[float]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sigma, _, _ = gen_sigma_y()
    prec = np.linalg.inv(sigma)
    return np.linalg.eigvalsh(cov_to_corr(prec)).tolist()


# ── Single-level explicit ───────────────────────────────────────────────────

def singlelevel_sigma_observed(seed: int, num_observed: int = 6, degree: float = 2.0) -> np.ndarray | None:
    """One hidden root node, num_observed observed nodes, simple random DAG."""
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            subview = gen_partially_observed(
                size_micro_node_dag=None,
                min_num_local_nodes=num_observed,
                max_num_local_nodes=num_observed,
                num_macro_nodes=1,          # single macro node = flat single-level DAG
                degree=degree,
                list_confounder2hide=[0.0],  # hide root cluster
                num_sample=10,               # minimal; we use analytical Σ
                output_dir=str(Path("/tmp") / f"singlelevel_seed{seed}"),
                rng=seed,
                plot=False,
                strategy_cls=RootConfounderDAG,
            )
        cov = np.cov(subview.data.T)
        return cov
    except Exception:
        return None


def singlelevel_eigenvalues_one(seed: int, num_observed: int = 6, degree: float = 2.0):
    cov = singlelevel_sigma_observed(seed, num_observed, degree)
    if cov is None or cov.shape[0] < 2:
        return None
    try:
        prec = np.linalg.inv(cov)
        return np.linalg.eigvalsh(cov_to_corr(prec)).tolist()
    except np.linalg.LinAlgError:
        return None


# ── Two-level explicit ──────────────────────────────────────────────────────

def twolevel_eigenvalues_one(seed: int, num_macro: int = 3, degree: float = 2.0,
                              min_micro: int = 2, max_micro: int = 4):
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            subview = gen_partially_observed(
                size_micro_node_dag=None,
                min_num_local_nodes=min_micro,
                max_num_local_nodes=max_micro,
                num_macro_nodes=num_macro,
                degree=degree,
                list_confounder2hide=[0.0],
                num_sample=10,
                output_dir=str(Path("/tmp") / f"twolevel_seed{seed}"),
                rng=seed,
                plot=False,
                strategy_cls=RootConfounderDAG,
            )
        cov = np.cov(subview.data.T)
        if cov.shape[0] < 2:
            return None
        prec = np.linalg.inv(cov)
        return np.linalg.eigvalsh(cov_to_corr(prec)).tolist()
    except Exception:
        return None


# ── Plotting ────────────────────────────────────────────────────────────────

def spectral_range(eigvals: list[float]) -> float:
    return max(eigvals) - min(eigvals)


def plot_and_save(implicit_eigs, single_eigs, two_eigs, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: eigenvalue distributions (violin)
    ax = axes[0]
    parts = ax.violinplot(
        [implicit_eigs, single_eigs, two_eigs],
        positions=[1, 2, 3], showmedians=True, showextrema=True,
    )
    colours = ["#9ecae1", "#a1d99b", "#fdae6b"]
    for body, c in zip(parts["bodies"], colours):
        body.set_facecolor(c)
        body.set_alpha(0.8)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Implicit\n(diag-dom Ω)", "Single-level\nexplicit", "Two-level\nexplicit"])
    ax.set_ylabel("Eigenvalue of normalized precision")
    ax.set_title("Eigenvalue distributions", fontsize=11)

    # Right: spectral range distributions
    ax = axes[1]
    imp_ranges = [spectral_range(e) for e in implicit_eigs_per_instance]
    sin_ranges = [spectral_range(e) for e in single_eigs_per_instance]
    two_ranges = [spectral_range(e) for e in two_eigs_per_instance]

    for ranges, label, c in zip(
        [imp_ranges, sin_ranges, two_ranges],
        ["Implicit", "Single-level", "Two-level"],
        ["#9ecae1", "#a1d99b", "#fdae6b"],
    ):
        ax.hist(ranges, bins=20, alpha=0.7, label=label, color=c, density=True)

    # KW test
    kw_stat, kw_p = stats.kruskal(imp_ranges, sin_ranges, two_ranges)
    ax.set_xlabel("Spectral range (λ_max − λ_min)")
    ax.set_ylabel("Density")
    ax.set_title(f"Spectral range distributions\n(Kruskal-Wallis p={kw_p:.4f})", fontsize=11)
    ax.legend()

    fig.suptitle(
        "Two-level block-hierarchical DGP vs alternatives\n"
        "Spectral coverage of the normalized precision matrix",
        fontsize=12,
    )
    fig.tight_layout()
    out_pdf = out_dir / "spectrum_3way_comparison.pdf"
    out_png = out_dir / "spectrum_3way_comparison.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    return out_png


def print_summary(imp_ranges, sin_ranges, two_ranges, n):
    print(f"\n=== Three-way spectral comparison ({n} instances each) ===")
    for name, ranges in [("Implicit", imp_ranges),
                         ("Single-level explicit", sin_ranges),
                         ("Two-level explicit", two_ranges)]:
        arr = np.array(ranges)
        print(f"  {name:25s}: mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")
    kw_stat, kw_p = stats.kruskal(imp_ranges, sin_ranges, two_ranges)
    print(f"\n  Kruskal-Wallis across 3 methods: p = {kw_p:.6f}")
    t_si, p_si = stats.mannwhitneyu(sin_ranges, imp_ranges, alternative="greater")
    t_ti, p_ti = stats.mannwhitneyu(two_ranges, imp_ranges, alternative="greater")
    t_ts, p_ts = stats.mannwhitneyu(two_ranges, sin_ranges, alternative="greater")
    print(f"  Mann-Whitney (single > implicit):  p = {p_si:.6f}")
    print(f"  Mann-Whitney (two   > implicit):   p = {p_ti:.6f}")
    print(f"  Mann-Whitney (two   > single):     p = {p_ts:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--num-macro", type=int, default=3)
    ap.add_argument("--min-micro", type=int, default=2)
    ap.add_argument("--max-micro", type=int, default=4)
    ap.add_argument("--degree", type=float, default=2.0)
    ap.add_argument("--num-observed-single", type=int, default=6,
                    help="Observed nodes for single-level comparison (≈ num_macro * avg_micro)")
    ap.add_argument("--output-dir", type=Path, default=Path("output/spectrum_3way"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n = args.trials
    print(f"Running {n} instances per method ...")

    global implicit_eigs_per_instance, single_eigs_per_instance, two_eigs_per_instance

    # Implicit
    print("  Implicit ...")
    implicit_eigs_per_instance = [implicit_eigenvalues_one() for _ in range(n)]

    # Single-level
    print("  Single-level explicit ...")
    single_eigs_per_instance = []
    seed = 0
    while len(single_eigs_per_instance) < n:
        e = singlelevel_eigenvalues_one(seed, args.num_observed_single, args.degree)
        if e:
            single_eigs_per_instance.append(e)
        seed += 1

    # Two-level
    print("  Two-level explicit ...")
    two_eigs_per_instance = []
    seed = 0
    while len(two_eigs_per_instance) < n:
        e = twolevel_eigenvalues_one(seed, args.num_macro, args.degree, args.min_micro, args.max_micro)
        if e:
            two_eigs_per_instance.append(e)
        seed += 1

    # Flatten eigenvalues for violin plot
    implicit_eigs = [v for ev in implicit_eigs_per_instance for v in ev]
    single_eigs   = [v for ev in single_eigs_per_instance  for v in ev]
    two_eigs      = [v for ev in two_eigs_per_instance      for v in ev]

    imp_ranges = [spectral_range(e) for e in implicit_eigs_per_instance]
    sin_ranges = [spectral_range(e) for e in single_eigs_per_instance]
    two_ranges = [spectral_range(e) for e in two_eigs_per_instance]

    print_summary(imp_ranges, sin_ranges, two_ranges, n)
    out_png = plot_and_save(implicit_eigs, single_eigs, two_eigs, args.output_dir)
    df_path = args.output_dir / "results_3way.csv"
    import pandas as pd
    pd.DataFrame({
        "method": ["implicit"] * n + ["single_level"] * n + ["two_level"] * n,
        "spectral_range": imp_ranges + sin_ranges + two_ranges,
    }).to_csv(df_path, index=False)
    print(f"Saved CSV → {df_path}")
    return out_png


if __name__ == "__main__":
    main()
