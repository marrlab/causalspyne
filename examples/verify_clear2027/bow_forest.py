"""Forest plot of the paired difference bows (B) minus bow-free (A) from bow_all_E.npz (bow_vs_bowfree_all.py).
Top row: SHD (FCI: normalized SHD to the exact PAG, in points; RCD, ICA-LiNGAM, ngBap: typed-edge SHD, in edges).
Bottom row: ancestral binary accuracy (points). Rows: weight range of |W_ij|; 95% percentile bootstrap intervals over the instances;
a star marks significance at 0.05 after Holm correction over the two settings (per algorithm and metric).
Usage: python bow_forest.py bow_all_E.npz out.png"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
sys.path.insert(0, ".")
import bow_vs_bowfree_all as b
d = np.load(sys.argv[1]); keys = list(d.files); names = b.column_names()
col = {n: i for i, n in enumerate(names)}
algos = ["FCI", "RCD", "ICA", "ngBap"]
shd_col = {"FCI": ("FCI PAG SHD", 100.0, "points"), "RCD": ("RCD typed SHD", 1.0, "edges"), "ICA": ("ICA typed SHD", 1.0, "edges"), "ngBap": ("ngBap typed SHD", 1.0, "edges")}
acc_col = {a: (f"{a} anc binary acc", 100.0, "points") for a in algos}
def holm(p):
    o = np.argsort(p); a = np.empty(len(p)); m = 0
    for r, i in enumerate(o): m = max(m, (len(p) - r) * p[i]); a[i] = min(m, 1)
    return a
rng = np.random.default_rng(0)
plt.rcParams["font.size"] = 7
fig, axs = plt.subplots(2, 4, figsize=(5.2, 2.6), sharey=True)
lab = lambda k: "0.1–0.5" if "0.1,0.5" in k else "0.5–1.5"
for r, spec in enumerate((shd_col, acc_col)):
    for c, a in enumerate(algos):
        ax = axs[r, c]; nm, scale, unit = spec[a]; q = col[nm]
        mu, lo, hi, ps = [], [], [], []
        for k in keys:
            R = d[k]; x = (R[:, 1, q] - R[:, 0, q]) * scale; ok = ~np.isnan(x); x = x[ok]
            bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
            mu.append(x.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
            ps.append(wilcoxon(R[ok, 0, q], R[ok, 1, q]).pvalue if np.any(R[ok, 0, q] != R[ok, 1, q]) else 1.0)
        adj = holm(np.array(ps)); y = np.arange(len(keys))[::-1]
        for yi, m, l, h, p in zip(y, mu, lo, hi, adj):
            colr = "#c0392b" if ((m > 0) == (r == 0)) else "#2a6fb0"      # red = worse with bows
            ax.errorbar(m, yi, xerr=[[m - l], [h - m]], fmt="o", color=colr, capsize=3)
            ax.text(1.02, yi, "*" if p < 0.05 else "n.s.", transform=ax.get_yaxis_transform(), va="center", fontsize=6)
        ax.axvline(0, color="k", lw=0.8); ax.grid(axis="x", alpha=0.3); ax.set_ylim(-0.6, len(keys) - 0.4)
        ax.set_xlabel(f"Δ ({unit})")
        if r == 0: ax.set_title(a if a != "ICA" else "ICA-LiNGAM", fontsize=8)
axs[0, 0].set_yticks(range(len(keys))); axs[0, 0].set_yticklabels([lab(k) for k in keys][::-1], fontsize=6.5)
axs[1, 0].set_yticks(range(len(keys))); axs[1, 0].set_yticklabels([lab(k) for k in keys][::-1], fontsize=6.5)
axs[0, 0].set_ylabel("SHD\n|W| range", fontsize=6.5); axs[1, 0].set_ylabel("ancestral acc.\n|W| range", fontsize=6.5)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=170)
print("saved", sys.argv[2])
