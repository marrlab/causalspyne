"""Usage: python avg_partial_corr.py [out.png]  (plot: mean of rho^2 per pair by |V|, 95% bootstrap intervals, one panel per weight range).
Average partial correlations per class: for the same random instances as prop5_check.py (DD, CRE with unequal loadings, general PD Omega), the per-pair mean of rho_ij^2 and of |rho_ij|,
by |V| and weight range, with paired differences CRE - DD and PD - DD (same W and bidirected pairs, same theta)."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from causalspyne.explicit_confounding_gen import random_instance, pair_loadings, omega_from_loadings
from scipy.stats import wilcoxon
rng = np.random.default_rng(11)
def pc(W, Om):
    n = len(W); A = np.linalg.inv(np.eye(n) - W); S = A @ Om @ A.T; P = np.linalg.inv(S); d = 1 / np.sqrt(np.diag(P)); Pn = d[:, None] * P * d[None, :]
    off = ~np.eye(n, dtype=bool); return (Pn[off] ** 2).mean(), np.abs(Pn[off]).mean()
res = {}
for n in (5, 10, 20, 30):
    for (wlo, whi) in ((0.1, 0.5), (0.5, 1.5)):
        rows = []
        for t in range(150):
            W, pairs = random_instance(rng, n, wlo, whi, p_edge=min(0.3, 3.0 / n)); k = len(pairs)
            if k == 0: continue
            th = rng.uniform(-2, 2, k)
            Om = {"DD": omega_from_loadings(pair_loadings(th, pairs, n)), "CRE": omega_from_loadings(pair_loadings(th, pairs, n, log_ratio=rng.uniform(-2, 2, k)))}
            L = rng.standard_normal((n, n)) * rng.uniform(0.2, 1.5); Om["PD"] = L @ L.T + np.eye(n)
            rows.append([pc(W, Om[c]) for c in ("DD", "CRE", "PD")])
        R = np.array(rows)                                         # instances x class x (mean rho^2, mean |rho|)
        res[(n, wlo, whi)] = R
print("per-pair mean of rho^2 (and of |rho|) by class; CRE-DD and PD-DD paired differences with Wilcoxon p")
for key, R in res.items():
    m = R.mean(0); p1 = wilcoxon(R[:, 0, 0], R[:, 1, 0]).pvalue; p2 = wilcoxon(R[:, 0, 0], R[:, 2, 0]).pvalue
    print(f"n={key[0]:2d} |W| in [{key[1]},{key[2]}]  DD {m[0,0]:.4f} ({m[0,1]:.3f})  CRE {m[1,0]:.4f} ({m[1,1]:.3f})  PD {m[2,0]:.4f} ({m[2,1]:.3f})   CRE-DD p={p1:.1e}  PD-DD p={p2:.1e}")
allR = np.concatenate(list(res.values())); print("overall means rho^2: DD %.4f CRE %.4f PD %.4f" % tuple(allR[:, :, 0].mean(0)))
print("share of instances with CRE > DD:", round(float((allR[:,1,0] > allR[:,0,0]).mean()),3), " PD > DD:", round(float((allR[:,2,0] > allR[:,0,0]).mean()),3))

if len(sys.argv) > 1:
    plt.rcParams["font.size"] = 7; brng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.1), sharey=True)
    col = {"DD": "#e8683a", "CRE": "#4fa36b", "PD": "#2a6fb0"}; names = {"DD": "diagonally dominant", "CRE": "canonical root explicit", "PD": "general PD"}
    for ax, (wlo, whi) in zip(axs, ((0.1, 0.5), (0.5, 1.5))):
        for ci, c in enumerate(("DD", "CRE", "PD")):
            xs, mu, lo, hi = [], [], [], []
            for n in (5, 10, 20, 30):
                x = res[(n, wlo, whi)][:, ci, 0]; bs = [brng.choice(x, len(x)).mean() for _ in range(2000)]
                xs.append(n * (1 + 0.04 * (ci - 1))); mu.append(x.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
            ax.errorbar(xs, mu, yerr=[np.array(mu) - lo, np.array(hi) - mu], fmt="o-", ms=3, lw=1, capsize=2, color=col[c], label=names[c])
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks([5, 10, 20, 30]); ax.set_xticklabels(["5", "10", "20", "30"]); ax.set_xlabel("$|V|$"); ax.set_title(f"$|W_{{i,j}}|$ in [{wlo}, {whi}]", fontsize=7); ax.grid(alpha=0.3)
    axs[0].set_ylabel("mean squared partial correlation\n(per pair, 150 instances)"); axs[0].legend(fontsize=5.5)
    fig.tight_layout(); fig.savefig(sys.argv[1], dpi=170)
