"""Forest plot of the paired SHD difference (RE - DD) of FCI, RCD and ICA-LiNGAM on non-Gaussian data, 8 settings,
100 ADMGs each (design E). Reads rcd_E.npz (RCD SHD idx 2, FCI SHD idx 7) and ica_E.npz (ICA SHD idx 2).
Usage: python shd_forest.py rcd_E.npz ica_E.npz out.png"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
rcd, ica, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
keys = list(rcd.files)
def holm(p):
    o = np.argsort(p); a = np.empty(len(p)); m = 0
    for r, i in enumerate(o):
        m = max(m, (len(p) - r) * p[i]); a[i] = min(m, 1)
    return a
algs = [("FCI", rcd, 7), ("RCD", rcd, 2), ("ICA-LiNGAM", ica, 2)]
def lab(k):
    w = "[0.1,0.5]" if "0.1,0.5" in k else "[0.5,1.5]"
    c = k.split("_c")[1].split("_")[0]; n = k.split("_N")[1]
    return f"|W|∈{w}  |Ω_ij|≤{c}  N={n}"
rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, (nm, d, q) in zip(axs, algs):
    mu, lo, hi, ps = [], [], [], []
    for k in keys:
        R = d[k]; x = (R[:, 1, q] - R[:, 0, q]) * 100
        bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        mu.append(x.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        ps.append(wilcoxon(R[:, 0, q], R[:, 1, q]).pvalue if np.any(R[:, 0, q] != R[:, 1, q]) else 1.0)
    adj = holm(np.array(ps)); y = np.arange(len(keys))[::-1]
    col = ["#c0392b" if m > 0 else "#2a6fb0" for m in mu]
    for yi, m, l, h, c, a in zip(y, mu, lo, hi, col, adj):
        ax.errorbar(m, yi, xerr=[[m - l], [h - m]], fmt="o", color=c, capsize=3)
        ax.text(1.02, yi, "*" if a < 0.05 else "n.s.", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.axvline(0, color="k", lw=0.8); ax.grid(axis="x", alpha=0.3)
    ax.set_title(nm); ax.set_xlabel("change in SHD: RE − DD  (points, 95% CI)")
axs[0].set_yticks(range(len(keys))); axs[0].set_yticklabels([lab(k) for k in keys][::-1], fontsize=8)
fig.suptitle("non-Gaussian data, paired per ADMG (100); right of 0 = worse under RE; * = significant after Holm correction", fontsize=9)
fig.tight_layout(); fig.savefig(out, dpi=170)
for nm, d, q in algs:
    print(nm, [round(float(((d[k][:, 1, q] - d[k][:, 0, q]) * 100).mean()), 1) for k in keys])
