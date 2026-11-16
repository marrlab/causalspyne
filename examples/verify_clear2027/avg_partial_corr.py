"""Average partial correlations per class: for the same random instances as prop5_check.py (DD, CRE with unequal loadings, general PD Omega), the per-pair mean of rho_ij^2 and of |rho_ij|,
by |V| and weight range, with paired differences CRE - DD and PD - DD (same W and bidirected pairs, same theta)."""
import numpy as np
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
