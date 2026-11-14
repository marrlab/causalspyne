"""Figure from ci_auc.npz: paired difference (PD - DD) per ADMG of CI-test recall and AUC, N=500.
Usage: python ci_auc_plot.py [npz] [out.pdf]"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

npz = sys.argv[1] if len(sys.argv) > 1 else "ci_auc.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "ci_test_performance.pdf"
D = np.load(npz)
sets = D["sets"]
INK, COL = "#3d3d3a", "#2a78d6"
fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
for ax, (m, title) in zip(axes, [(0, "recall of the dependent assertions"), (2, "AUC of $|z|$ (dependent vs independent)")]):
    for si in range(len(sets)):
        M = D[f"s{si}"]
        d = M[:, 1, m] - M[:, 0, m]
        bp = ax.boxplot(d[~np.isnan(d)], positions=[si], widths=0.45, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": INK, "lw": 0.8},
                        capprops={"color": INK, "lw": 0.8}, boxprops={"edgecolor": INK, "lw": 0.8})
        bp["boxes"][0].set_facecolor(COL)
    ax.axhline(0, color=INK, lw=0.8, ls="--")
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([f"$|W|\\in[{a},{b}]$\n$c={c}$" for a, b, c in sets], fontsize=8)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_ylabel("PD $-$ DD (paired, per ADMG)", fontsize=9)
    ax.grid(axis="y", color="#e4e2dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
print("saved", out)
