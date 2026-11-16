"""Five ways to present the DD vs PD∖DD CI-test result (reads ci_present.npz).  Usage: python ci_present_plot.py outdir"""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
D = np.load("ci_present.npz"); NS = list(D["NS"]); I500 = NS.index(500); SETS = D["SETS"]
FAM = {"G": "Gaussian: explicit extreme vs DD", "N": "non-Gaussian: canonical explicit vs DD"}
rng = np.random.default_rng(0)
ORDER = [(f, s) for f in "GN" for s in range(4)]
def lab(f, s):
    w0, w1, c = SETS[s]; return f"{'G' if f=='G' else 'U'}  {w0}–{w1}  ≤{c}"   # noise (G Gaussian, U uniform), range of |W|, bound on |Omega_ij|
BLUE, ORANGE = "#2a6fb0", "#d9731a"

def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1, run)
    return adj

# 1. forest plot ------------------------------------------------------------------------------------------------
plt.rcParams["font.size"] = 7
fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.9), sharey=True)
pv = {m: [] for m in (0, 1)}
for f, s in ORDER:
    M = D[f"{f}{s}_metrics"][:, :, I500, :]            # R x arm x metric
    for m in (0, 1):
        d = M[:, 1, m] - M[:, 0, m]; d = d[~np.isnan(d)]
        pv[m].append(wilcoxon(d).pvalue)
for m, ax in enumerate(axs):
    adj = holm(np.array(pv[m]))
    for k, (f, s) in enumerate(ORDER):
        M = D[f"{f}{s}_metrics"][:, :, I500, :]
        d = (M[:, 1, m] - M[:, 0, m]); d = d[~np.isnan(d)] * 100
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        y = len(ORDER) - 1 - k
        ax.errorbar(d.mean(), y, xerr=[[d.mean() - np.percentile(bs, 2.5)], [np.percentile(bs, 97.5) - d.mean()]],
                    fmt="o", color=BLUE if f == "G" else ORANGE, capsize=3)
        ax.text(1.02, y, "*" if adj[k] < 0.05 else "n.s.", transform=ax.get_yaxis_transform(), va="center", fontsize=6.5)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel(f"Δ {['recall','AUC'][m]} (points)")
    ax.set_title(["CI-test recall", "AUC of |z|"][m], fontsize=8); ax.grid(axis="x", alpha=0.3)
axs[0].set_yticks(range(len(ORDER))); axs[0].set_yticklabels([lab(f, s) for f, s in ORDER][::-1], fontsize=6.5)
axs[0].set_ylabel("noise, |W| range, bound on |Ω_ij|", fontsize=6)
# (title dropped: N=500, 100 ADMGs and the star legend are in the paper caption)
fig.tight_layout(); fig.savefig(f"{out}/present1_forest.png", dpi=170); plt.close(fig)

# 2. recall vs true partial correlation strength -----------------------------------------------------------------
edges = np.array([0, 0.03, 0.06, 0.1, 0.2, 0.35, 1.01]); cen = [f"{a:g}–{b:g}" for a, b in zip(edges[:-1], np.minimum(edges[1:], 1))]
sel = [("G", 1), ("G", 3), ("N", 1), ("N", 3)]
fig, axs = plt.subplots(2, 4, figsize=(13, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
from ci_test_performance import Z
for j, (f, s) in enumerate(sel):
    for a_, col, nm in ((0, BLUE, "DD"), (1, ORANGE, "PD∖DD")):
        z, pc, dep = D[f"{f}{s}_raw{a_}"]; dep = dep.astype(bool)
        rec, frac = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = dep & (pc >= lo) & (pc < hi)
            rec.append((z[m] > Z).mean() if m.sum() > 20 else np.nan); frac.append(m.sum() / dep.sum())
        axs[0, j].plot(range(len(cen)), rec, "o-", color=col, label=nm)
        axs[1, j].bar(np.arange(len(cen)) + (0.2 if a_ else -0.2), frac, 0.4, color=col)
    axs[0, j].set_title(lab(f, s), fontsize=9); axs[0, j].set_ylim(0, 1.02); axs[0, j].grid(alpha=0.3)
    axs[1, j].set_xticks(range(len(cen))); axs[1, j].set_xticklabels(cen, rotation=40, fontsize=7)
    axs[1, j].set_xlabel("true |partial correlation| of the assertion")
axs[0, 0].set_ylabel("recall of the CI test"); axs[1, 0].set_ylabel("share of dependent\nassertions"); axs[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{out}/present2_recall_vs_strength.png", dpi=170); plt.close(fig)

# 3. sample-size equivalent --------------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
fac = []
for k, (f, s) in enumerate(ORDER):
    M = D[f"{f}{s}_metrics"][:, :, :, 0].mean(0)       # arm x N recall
    target = M[0, I500]
    big = np.array(NS)[M[1] >= target]
    nstar = big.min() if len(big) else np.nan
    if len(big) and big.min() > NS[0]:
        j = NS.index(big.min()); n0, n1, r0, r1 = NS[j - 1], NS[j], M[1, j - 1], M[1, j]
        nstar = np.exp(np.log(n0) + (target - r0) / (r1 - r0) * (np.log(n1) - np.log(n0)))
    fac.append(nstar / 500)
    if (f, s) in [("G", 3), ("N", 3)]:
        for a_, col, nm in ((0, BLUE, "DD"), (1, ORANGE, "PD∖DD")):
            ax1.plot(NS, M[a_], "o-" if f == "G" else "s--", color=col, ms=3, label=f"{nm} ({'Gauss' if f=='G' else 'non-G'})")
ax1.set_xscale("log"); ax1.set_xlabel("sample size N"); ax1.set_ylabel("CI-test recall"); ax1.grid(alpha=0.3); ax1.legend(fontsize=7)
ax1.set_title("recall vs N, |W|∈[0.5,1.5], |Ω_ij|≤2", fontsize=9)
cens = [np.isnan(x) for x in fac]; fac = [10.0 if np.isnan(x) else x for x in fac]
ax2.barh(range(len(ORDER)), fac[::-1], color=[BLUE if f == "G" else ORANGE for f, s in ORDER][::-1])
for k_, c_ in enumerate(cens[::-1]):
    if c_: ax2.text(5, k_, 'not reached by N=5000 (>10x)', va='center', ha='center', color='w', fontsize=7)
ax2.set_yticks(range(len(ORDER))); ax2.set_yticklabels([lab(f, s) for f, s in ORDER][::-1], fontsize=8)
ax2.axvline(1, color="k", lw=0.8); ax2.set_xlabel("samples the PD∖DD class needs, relative to DD at N=500")
ax2.set_title("sample-size equivalent", fontsize=9)
fig.tight_layout(); fig.savefig(f"{out}/present3_sample_size.png", dpi=170); plt.close(fig)
np.savetxt(f"{out}/present3_factors.txt", np.array(fac), fmt="%.2f")

# 4. pooled ROC ---------------------------------------------------------------------------------------------------
def roc(z, dep):
    o = np.argsort(-z); d = dep[o]; tp = np.cumsum(d) / d.sum(); fp = np.cumsum(~d) / (~d).sum(); return fp, tp
fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.3), sharey=True)
for j, (f, s) in enumerate(sel):
    for a_, col, nm in ((0, BLUE, "DD"), (1, ORANGE, "PD∖DD")):
        z, pc, dep = D[f"{f}{s}_raw{a_}"]; dep = dep.astype(bool)
        fp, tp = roc(z, dep); a = np.trapz(tp, fp)
        axs[j].plot(fp, tp, color=col, label=f"{nm}, AUC {a:.3f}")
    axs[j].plot([0, 1], [0, 1], "k:", lw=0.6); axs[j].set_title(lab(f, s), fontsize=9); axs[j].set_xlabel("false positive rate"); axs[j].legend(fontsize=7, loc="lower right")
axs[0].set_ylabel("true positive rate (pooled over 100 ADMGs)")
fig.tight_layout(); fig.savefig(f"{out}/present4_roc.png", dpi=170); plt.close(fig)

# 5. paired scatter -----------------------------------------------------------------------------------------------
fig, axs = plt.subplots(2, 4, figsize=(12.5, 6))
for ax, (f, s) in zip(axs.ravel(), ORDER):
    M = D[f"{f}{s}_metrics"][:, :, I500, 0]
    ax.scatter(M[:, 0], M[:, 1], s=10, alpha=0.6, color=BLUE if f == "G" else ORANGE)
    lo = min(M.min(), 1) - 0.0; ax.plot([M.min(), 1], [M.min(), 1], "k-", lw=0.7)
    ax.set_title(lab(f, s), fontsize=8); ax.set_xlabel("recall, DD"); ax.set_ylabel("recall, PD∖DD")
    ax.text(0.04, 0.9, f"below diagonal in {(M[:,1] < M[:,0]).mean()*100:.0f}% of ADMGs", transform=ax.transAxes, fontsize=7)
fig.tight_layout(); fig.savefig(f"{out}/present5_paired_scatter.png", dpi=170); plt.close(fig)
print("ok")
