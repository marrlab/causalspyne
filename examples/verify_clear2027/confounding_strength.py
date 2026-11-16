"""Diagonal dominance caps confounding strength (figure for Appendix F.1).

One confounder with |loadings| in [1, 2] on k of n=9 observables, unit noise.
Per instance both arms share W, the confounded set and the off-diagonal Omega:
  explicit : Omega = Lambda Lambda^T + I
  diag-dom : same off-diagonals, diagonal = 1 + row sum of |off-diagonal|
Panels: (a) rho(R_eps), (b) strongest correlation the confounder induces between
two observables (W = 0), (c) largest rho(R~) reachable with a FIXED W per class (medians over 24 W, read from
fixedW_three_classes_output.txt, produced by fixedW_three_classes.py). The earlier panel (c) with a freshly
drawn random W was dropped on 2026-11-13 because that comparison is not fair.
Medians and 10-90% bands over the trials.
"""
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_EXP, C_DD, INK, GRID = "#2a78d6", "#eb6834", "#3d3d3a", "#e4e2dc"


def corr(M):
    d = 1 / np.sqrt(np.diag(M))
    return d[:, None] * M * d[None, :]


def random_dag(rng, n, p, lo, hi):
    W = np.zeros((n, n))
    mask = np.tril(rng.random((n, n)) < p, -1)
    W[mask] = rng.uniform(lo, hi, mask.sum()) * rng.choice([-1, 1], mask.sum())
    perm = rng.permutation(n)
    return W[np.ix_(perm, perm)]


def run(trials, n, seed):
    rng = np.random.default_rng(seed)
    ks = list(range(2, n + 1))
    out = {m: {a: [] for a in ("exp", "dd")} for m in ("reps", "corr")}
    off = ~np.eye(n, dtype=bool)
    for k in ks:
        acc = {m: {a: [] for a in ("exp", "dd")} for m in out}
        for _ in range(trials):
            lam = np.zeros(n)
            ch = rng.choice(n, k, replace=False)
            lam[ch] = rng.uniform(1, 2, k) * rng.choice([-1, 1], k)
            Oe = np.outer(lam, lam) + np.eye(n)
            O = Oe - np.diag(np.diag(Oe))
            Od = O + np.diag(1 + np.abs(O).sum(1))
            for a, Om in (("exp", Oe), ("dd", Od)):
                acc["reps"][a].append(np.linalg.eigvalsh(corr(Om))[-1])
                acc["corr"][a].append(np.abs(corr(Om)[off]).max())  # W = 0: Sigma = Omega
        for m in out:
            for a in ("exp", "dd"):
                v = np.array(acc[m][a])
                out[m][a].append(np.percentile(v, [10, 50, 90]))
    return ks, {m: {a: np.array(out[m][a]) for a in out[m]} for m in out}


def read_extremes(path):
    rows = []
    for line in open(path):
        m = re.match(r"\|W\| in \[([\d.]+),([\d.]+)\], c=([\d.]+)", line)
        if m:
            rho = [float(x) for x in re.findall(r"(?:none|DD|SDD|PD): rho ([\d.]+)", line)]
            rows.append((f"[{m.group(1)},{m.group(2)}]\n$|\\Omega_{{ij}}|\\leq{m.group(3)}$", rho))
    return rows


def plot(ks, res, path, extremes=None, panels="ac"):
    """panels: letters of the panels to draw, in order. a = rho(R_eps) vs k, b = strongest induced correlation
    with W = 0 (dropped from the main text on 2026-11-15: not claimed there), c = fixed-W extremes.
    Panels are labelled (a), (b), ... in the order drawn."""
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + (0 if n > 2 else 0.2), 3.1))
    axes = np.atleast_1d(axes)
    spec = {"a": ("reps", "noise correlation matrix", r"spectral radius $\rho(R_\epsilon)$"),
            "b": ("corr", "induced correlation, no directed edges ($W=0$)", r"largest $|\mathrm{corr}(Y_i,Y_j)|$")}
    for idx, (ax, p) in enumerate(zip(axes, panels)):
        letter = "(" + "abc"[idx] + ") "
        if p in "ab":
            m, t, yl = spec[p]
            for arm, c, mk, lab in (("exp", C_EXP, "o", "explicit  $\\Lambda\\Lambda^T+I$"),
                                    ("dd", C_DD, "s", "diagonally dominant")):
                q = res[m][arm]
                ax.fill_between(ks, q[:, 0], q[:, 2], color=c, alpha=0.15, linewidth=0)
                ax.plot(ks, q[:, 1], color=c, lw=2, marker=mk, ms=5, label=lab)
            if m == "reps":
                ax.axhline(2, color=INK, lw=1, ls="--")
                ax.text(ks[-1], 2.15, "bound 2", ha="right", va="bottom", color=INK, fontsize=8)
                ax.legend(frameon=False, fontsize=8, loc="upper left")
            ax.set_title(letter + t, fontsize=9, color=INK)
            ax.set_ylabel(yl)
            ax.set_xlabel("observables affected by the confounder")
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        else:
            rows = read_extremes(extremes)
            cols = [("#b9b7ae", "no confounding"), (C_DD, "diagonally dominant"), ("#4fa36b", "canonical explicit"), (C_EXP, "general PD")]
            w = 0.2
            for q, (c, lab) in enumerate(cols):
                ax.bar(np.arange(len(rows)) + (q - 1.5) * w, [r[1][q] for r in rows], w, color=c, label=lab)
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels([r[0] for r in rows], fontsize=7)
            ax.set_xlabel(r"$|W_{ij}|$ range and entry bound on $|\Omega_{ij}|$")
            ax.set_title(letter + r"partial correlation matrix, fixed $W$", fontsize=9, color=INK)
            ax.set_ylabel(r"largest spectral radius $\rho(\tilde{R})$")
            ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            ax.set_ylim(0, 7.4)
            ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="confounding_strength.pdf")
    ap.add_argument("--extremes", default="fixedW_three_classes_output.txt")
    ap.add_argument("--panels", default="ac", help="panels to draw; main text uses ac, abc gives the three-panel version")
    a = ap.parse_args()
    ks, res = run(a.trials, a.n, a.seed)
    for m in res:
        print(m, "k:", ks)
        for arm in res[m]:
            print("  ", arm, np.round(res[m][arm][:, 1], 2).tolist())
    plot(ks, res, a.out, a.extremes, a.panels)
    print("saved", a.out)
