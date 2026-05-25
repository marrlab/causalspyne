"""
Compare power spectrum (eigenvalue distribution of the normalized precision
matrix) between:

  - Implicit method: off-diagonal entries of Omega constrained by diagonal
    dominance → restricted spectral range (proved in theory section)
  - Explicit method: full DAG with hidden root cluster, covariance derived
    analytically from the SEM → wider spectral range

For each method we compute all eigenvalues of the correlation-normalized
precision matrix (i.e., the partial-correlation matrix eigenvalues) over
many random instances and plot the distributions side by side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from causalspyne.implicit_gen_Sigma import gen_sigma_y, cov_to_corr
from causalspyne.implicit_omega import get_extreme_eigenvalue
from causalspyne.dag_gen import GenDAG
from causalspyne.dag_gen_topo_order import RootConfounderDAG
from causalspyne.gen_dag_2level import GenDAG2Level
from causalspyne.utils_random import coerce_rng


# ---------------------------------------------------------------------------
# Implicit spectrum
# ---------------------------------------------------------------------------

def implicit_eigenvalues_one(suppress_print=True):
    """
    Return all eigenvalues of the normalized precision for one implicit sample.
    Suppresses the noisy print statements inside gen_sigma_y.
    """
    import io, contextlib
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if suppress_print else contextlib.nullcontext()
    with ctx:
        mat_sigma, _, _ = gen_sigma_y()
    inv_sigma = np.linalg.inv(mat_sigma)
    mat_precision_corr = cov_to_corr(inv_sigma)
    eigvals = np.linalg.eigvalsh(mat_precision_corr)
    return eigvals.tolist()


# ---------------------------------------------------------------------------
# Explicit spectrum
# ---------------------------------------------------------------------------

def explicit_sigma_observed(dag_gen_2level: GenDAG2Level) -> np.ndarray:
    """
    Analytically compute the covariance matrix of observed variables.

    For a linear Gaussian SEM x = W*x + e with e ~ N(0, I):
        Sigma_all = (I - W)^{-1} (I - W)^{-T}
    Sigma_observed = submatrix indexed by observed global indices.
    """
    W = dag_gen_2level.dag_refined.mat_adjacency  # weighted, mat[i,j] = coeff j->i
    n = W.shape[0]
    IminusW = np.eye(n) - W
    inv_IminusW = np.linalg.inv(IminusW)
    sigma_all = inv_IminusW @ inv_IminusW.T

    # observed indices = all indices minus root macro cluster indices
    root_names = dag_gen_2level.get_root_macro_names()
    hidden = []
    for name in root_names:
        hidden.extend(dag_gen_2level.get_macro_node_global_inds(name))
    all_inds = list(range(n))
    obs_inds = [i for i in all_inds if i not in hidden]

    sigma_obs = sigma_all[np.ix_(obs_inds, obs_inds)]
    return sigma_obs


def explicit_eigenvalues_one(rng, num_macro_nodes, size_micro, degree):
    """Return all eigenvalues of normalized precision for one explicit sample."""
    simple_dag_gen = GenDAG(
        num_nodes=size_micro, degree=degree, rng=rng,
        strategy_cls=RootConfounderDAG,
    )
    dag_gen = GenDAG2Level(
        dag_generator=simple_dag_gen,
        num_macro_nodes=num_macro_nodes,
        num_micro_nodes=size_micro,
        rng=rng,
    )
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dag_gen.run()

    sigma_obs = explicit_sigma_observed(dag_gen)
    if sigma_obs.shape[0] < 2:
        return []
    try:
        inv_sigma = np.linalg.inv(sigma_obs)
    except np.linalg.LinAlgError:
        return []
    mat_precision_corr = cov_to_corr(inv_sigma)
    eigvals = np.linalg.eigvalsh(mat_precision_corr)
    return eigvals.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_eigenvalues(n_trials, num_macro_nodes, size_micro, degree, seed):
    implicit_eigvals = []
    explicit_eigvals = []
    rng_base = np.random.default_rng(seed)

    for i in range(n_trials):
        try:
            implicit_eigvals.extend(implicit_eigenvalues_one())
        except Exception:
            pass

        child_seed = int(rng_base.integers(0, 2**31))
        rng = coerce_rng(child_seed)
        try:
            explicit_eigvals.extend(
                explicit_eigenvalues_one(rng, num_macro_nodes, size_micro, degree)
            )
        except Exception:
            pass

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_trials} trials done", flush=True)

    return np.array(implicit_eigvals), np.array(explicit_eigvals)


def plot_spectra(implicit_eigvals, explicit_eigvals, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, eigvals, label, color in zip(
        axes,
        [implicit_eigvals, explicit_eigvals],
        ["Implicit (diagonal-dominant Ω)\n[restricted spectrum]",
         "Explicit (DAG + hidden root)\n[wider spectrum]"],
        ["steelblue", "darkorange"],
    ):
        ax.hist(eigvals, bins=50, density=True, color=color, alpha=0.75)
        ax.axvline(np.median(eigvals), color="black", linestyle="--",
                   linewidth=1, label=f"median={np.median(eigvals):.2f}")
        ax.set_xlabel("Eigenvalue of normalized precision")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle("Power spectrum: implicit vs explicit unobserved confounding")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {output_path}")


def print_summary(implicit_eigvals, explicit_eigvals):
    print("\n=== Spectral summary ===")
    for name, ev in [("Implicit", implicit_eigvals), ("Explicit", explicit_eigvals)]:
        print(f"{name:10s}  n={len(ev):5d}  "
              f"min={ev.min():.3f}  max={ev.max():.3f}  "
              f"mean={ev.mean():.3f}  std={ev.std():.3f}  "
              f"range={ev.max()-ev.min():.3f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare power spectrum: implicit vs explicit confounding."
    )
    p.add_argument("--trials", type=int, default=200,
                   help="Number of random instances per method")
    p.add_argument("--num-macro-nodes", type=int, default=4)
    p.add_argument("--size-micro", type=int, default=3)
    p.add_argument("--degree", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("output/spectrum"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting eigenvalues over {args.trials} trials per method...")
    implicit_ev, explicit_ev = collect_eigenvalues(
        args.trials, args.num_macro_nodes, args.size_micro, args.degree, args.seed
    )

    print_summary(implicit_ev, explicit_ev)
    plot_spectra(implicit_ev, explicit_ev,
                 args.output_dir / "spectrum_comparison.pdf")

    np.save(args.output_dir / "implicit_eigvals.npy", implicit_ev)
    np.save(args.output_dir / "explicit_eigvals.npy", explicit_ev)


if __name__ == "__main__":
    main()
