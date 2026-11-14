"""Figure from ci_test_performance.npz: paired difference (PD - DD) of CI-test recall and accuracy per ADMG.
Usage: python ci_test_performance_plot.py [npz] [out.pdf]"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

npz = sys.argv[1] if len(sys.argv) > 1 else "ci_test_performance.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "ci_test_performance.pdf"
D = np.load(npz)
sets, ns = D["sets"], D["ns"]
INK, C500, C5000 = "#3d3d3a", "#2a78d6", "#eb6834"
fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
for ax, (m, title) in zip(axes, [(0, "recall of dependent assertions"), (2, "accuracy over all assertions")]):
    for si, (wlo, whi, c) in enumerate(sets):
        M = D[f"s{si}"]  # inst x N x arm x metric
        for bi, (N, col) in enumerate(zip(ns, (C500, C5000))):
            d = M[:, bi, 1, m] - M[:, bi, 0, m]
            pos = si + (bi - 0.5) * 0.36
            bp = ax.boxplot(d[~np.isnan(d)], positions=[pos], widths=0.3, patch_artist=True, showfliers=False,
                            medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": INK, "lw": 0.8},
                            capprops={"color": INK, "lw": 0.8}, boxprops={"edgecolor": INK, "lw": 0.8})
            bp["boxes"][0].set_facecolor(col)
    ax.axhline(0, color=INK, lw=0.8, ls="--")
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([f"$|W|\\in[{a},{b}]$\n$c={c}$" for a, b, c in sets], fontsize=8)
    ax.set_title(title, fontsize=9, color=INK)
    ax.grid(axis="y", color="#e4e2dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("PD $-$ DD (paired, per ADMG)", fontsize=9)
axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=C500, ec=INK), plt.Rectangle((0, 0), 1, 1, fc=C5000, ec=INK)],
               labels=[f"N={ns[0]}", f"N={ns[1]}"], fontsize=8, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
print("saved", out)
