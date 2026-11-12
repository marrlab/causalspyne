"""Partial-correlation spectrum under the diagonally dominant construction:
sparse bounded Omega (restricted) vs dense Omega (not restricted).

For |V| in a grid, draw a random DAG W (expected in-degree about 1, |w| in [0.1,0.5]) and
  sparse : each variable confounded with at most s others, |Omega_ij| in [0.1, c]
  dense  : every pair confounded, |Omega_ij| = c_dense
both completed by the paper's construction Omega_ii = 1 + sum_j |Omega_ij|.
Reports rho(R~) (largest eigenvalue of the normalized precision matrix) and the corollary
bound 2(1+s c)(1+d_in_max) for the sparse arm.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_SP, C_DE, INK, GRID = "#2a78d6", "#eb6834", "#3d3d3a", "#e4e2dc"


def rho_rtilde(W, Om):
    n = W.shape[0]
    A = np.eye(n) - W
    P = A.T @ np.linalg.inv(Om) @ A
    d = 1 / np.sqrt(np.diag(P))
    return np.linalg.eigvalsh(d[:, None] * P * d[None, :])[-1]


def random_dag(rng, n, exp_indeg):
    p = min(1.0, exp_indeg / max(1, (n - 1) / 2))
    W = np.zeros((n, n))
    m = np.tril(rng.random((n, n)) < p, -1)
    W[m] = rng.uniform(0.1, 0.5, m.sum()) * rng.choice([-1, 1], m.sum())
    perm = rng.permutation(n)
    return W[np.ix_(perm, perm)]


def sparse_offdiag(rng, n, s, c):
    O = np.zeros((n, n)); deg = np.zeros(n, int)
    for _ in range(n * s):
        i, j = rng.choice(n, 2, replace=False)
        if deg[i] < s and deg[j] < s and O[i, j] == 0:
            v = rng.uniform(0.1, c) * rng.choice([-1, 1])
            O[i, j] = O[j, i] = v; deg[i] += 1; deg[j] += 1
    return O


def complete(O):
    return O + np.diag(1 + np.abs(O).sum(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--s", type=int, default=2)
    ap.add_argument("--c", type=float, default=0.7)
    ap.add_argument("--c-dense", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rtilde_sparse_vs_dense.pdf")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    ns = [10, 20, 50, 100, 200]
    res = {"sparse": [], "dense": [], "bound": []}
    viol = 0
    for n in ns:
        rs, rd, bd = [], [], []
        for _ in range(a.trials):
            W = random_dag(rng, n, 1.0)
            d_in = int((W != 0).sum(1).max())
            Os = complete(sparse_offdiag(rng, n, a.s, a.c))
            Od = complete(-a.c_dense * (np.ones((n, n)) - np.eye(n)))
            r = rho_rtilde(W, Os); b = 2 * (1 + a.s * a.c) * (1 + d_in)
            viol += r > b + 1e-9
            rs.append(r); rd.append(rho_rtilde(W, Od)); bd.append(b)
        for k, v in (("sparse", rs), ("dense", rd), ("bound", bd)):
            res[k].append(np.percentile(v, [10, 50, 90, 100]))
        print(f"|V|={n:4d}  sparse median {np.median(rs):5.2f} max {max(rs):5.2f} | bound median {np.median(bd):5.2f} | dense median {np.median(rd):7.2f}")
    print("corollary bound violations:", viol)
    res = {k: np.array(v) for k, v in res.items()}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(ns, ns, color=INK, lw=1, ls=":", label=r"maximum possible, $|V|$")
    ax.plot(ns, res["bound"][:, 1], color=C_SP, lw=1, ls="--", label=r"bound $2(1+sc)(1+d^{\mathrm{in}}_{\max})$ (median)")
    for k, c, mk, lab in (("sparse", C_SP, "o", f"sparse $\\Omega$ ($s={a.s}$, $c={a.c}$)"),
                          ("dense", C_DE, "s", f"dense $\\Omega$ (all pairs, $c={a.c_dense}$)")):
        q = res[k]
        ax.fill_between(ns, q[:, 0], q[:, 2], color=c, alpha=0.15, linewidth=0)
        ax.plot(ns, q[:, 1], color=c, lw=2, marker=mk, ms=5, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"number of variables $|V|$"); ax.set_ylabel(r"$\rho(\tilde R)$")
    ax.grid(color=GRID, lw=0.8, which="major"); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
