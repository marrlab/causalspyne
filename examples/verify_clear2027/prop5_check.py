"""Numerical check and plot of Proposition 5: (1/n) sum_{i != j} rho_ij^2 <= rho(R~) - 1  (equivalently the mean squared partial correlation per pair is at most (rho(R~) - 1)/(n - 1)).
Instances: weighted DAG W (causalspyne.explicit_confounding_gen.random_instance, n = 5, 10, 20, 30; weight ranges [0.1,0.5] and [0.5,1.5]), bidirected pairs,
and Omega from three classes: DD (equal loadings), CRE (unequal loadings a/b = exp(2r), |r| <= 2), and general PD (Omega = L L^T + I with a dense random L).
Sigma = (I - W)^-1 Omega (I - W)^-T, P = normalized precision, R~ = -P. Usage: python prop5_check.py out.png   (prints the number of violations and the tightness)"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from causalspyne.explicit_confounding_gen import random_instance, pair_loadings, omega_from_loadings
rng = np.random.default_rng(11)
def stats(W, Om):
    n = len(W); A = np.linalg.inv(np.eye(n) - W); S = A @ Om @ A.T; P = np.linalg.inv(S); d = 1 / np.sqrt(np.diag(P)); Pn = d[:, None] * P * d[None, :]
    lhs = ((Pn ** 2).sum() - n) / n; rho = np.linalg.eigvalsh(Pn)[-1]
    return lhs, rho - 1
rows = {"DD": [], "CRE": [], "PD": []}; viol = 0; tot = 0
for n in (5, 10, 20, 30):
    for (wlo, whi) in ((0.1, 0.5), (0.5, 1.5)):
        for t in range(150):
            W, pairs = random_instance(rng, n, wlo, whi, p_edge=min(0.3, 3.0 / n))
            k = len(pairs)
            if k == 0: continue
            th = rng.uniform(-2, 2, k)
            Om = {"DD": omega_from_loadings(pair_loadings(th, pairs, n)),
                  "CRE": omega_from_loadings(pair_loadings(th, pairs, n, log_ratio=rng.uniform(-2, 2, k)))}
            L = rng.standard_normal((n, n)) * rng.uniform(0.2, 1.5); Om["PD"] = L @ L.T + np.eye(n)
            for cls, O in Om.items():
                m, b = stats(W, O); rows[cls].append((m, b, n)); tot += 1; viol += m > b + 1e-12
print(f"instances {tot}, violations of the inequality: {viol}")
for cls, v in rows.items():
    v = np.array(v); print(f"{cls}: bound/actual ratio median {np.median(v[:,1]/v[:,0]):.2f}, min {np.min(v[:,1]/v[:,0]):.2f}, max {np.max(v[:,1]/v[:,0]):.1f}")
plt.rcParams["font.size"] = 7
fig, ax = plt.subplots(figsize=(3.4, 2.6))
col = {"DD": "#e8683a", "CRE": "#4fa36b", "PD": "#2a6fb0"}
for cls, v in rows.items():
    v = np.array(v); ax.scatter(v[:, 1], v[:, 0], s=5, alpha=0.35, color=col[cls], label={"DD": "diagonally dominant", "CRE": "canonical root explicit", "PD": "general PD"}[cls])
lim = [min(np.array(rows['DD'])[:, 0].min(), 1e-4), 1.0]
ax.plot([1e-2, 30], [1e-2, 30], "k--", lw=0.8, label="bound (equality)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.3, 30); ax.set_ylim(0.1, 30)
ax.set_xlabel(r"bound $\rho(\tilde R)-1$"); ax.set_ylabel(r"$\frac{1}{|V|}\sum_{i\neq j}\rho_{i,j|\cdot}^2$"); ax.legend(fontsize=6, loc="lower right")
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=170)
