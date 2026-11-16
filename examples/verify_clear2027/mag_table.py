"""LaTeX rows of the appendix table 'RE - DD against the ADMG and against the exact MAG' from rcd_ica_mag_E.npz (Holm over the 8 settings per column)."""
import numpy as np
from scipy.stats import wilcoxon
d = np.load("rcd_ica_mag_E.npz"); keys = list(d.files)
cols = [2, 12, 7, 13]
def holm(p):
    o = np.argsort(p); a = np.empty(len(p)); m = 0
    for r, i in enumerate(o): m = max(m, (len(p) - r) * p[i]); a[i] = min(m, 1)
    return a
diff = {q: [] for q in cols}; adj = {}
for q in cols:
    ps = []
    for k in keys:
        R = d[k]; x = (R[:, 1, q] - R[:, 0, q]) * 100
        diff[q].append(x.mean()); ps.append(wilcoxon(R[:, 0, q], R[:, 1, q]).pvalue if np.any(R[:, 0, q] != R[:, 1, q]) else 1.0)
    adj[q] = holm(np.array(ps))
for i, k in enumerate(keys):
    w = "0.1--0.5" if "0.1,0.5" in k else "0.5--1.5"; c = k.split("_c")[1].split("_")[0]; n = k.split("_N")[1]
    cells = [f"${diff[q][i]:+.1f}${'$^\\star$' if adj[q][i] < 0.05 else ''}" for q in cols]
    print(f"{w} & {c} & {n} & " + " & ".join(cells) + r" \\")
