"""LaTeX rows of the appendix table 'CRE - DD for typed-edge SHD, binary ancestral accuracy and ancestral AUC' from ancestral_eval_E.npz (stored
estimates, see analyse_stored_estimates.py). Holm correction over the four settings, per algorithm and metric.
Usage: python ancestral_table.py > ancestral_table_rows.tex"""
import numpy as np
from scipy.stats import wilcoxon
import ancestral_eval as ae
from causalspyne.explicit_confounding_gen import random_instance
from causalspyne.pair_shd import typed_edge_shd
from causalspyne.ancestral_acc import ancestral_binary_accuracy
d = np.load("ancestral_eval_E.npz"); n = 10; names = ae.column_names()
settings = [(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)] for c in [0.7, 2.0] for N in [500, 2000]]
def holm(p):
    o = np.argsort(p); a = np.empty(len(p)); m = 0
    for r, i in enumerate(o): m = max(m, (len(p) - r) * p[i]); a[i] = min(m, 1)
    return a
keys = [k for k in d.files]
algos = ["RCD", "ICA", "FCI", "ngBap"]
vals = {}                                     # (algo, metric) -> list over settings of arrays (DD, CRE)
for key in keys:
    s = next(i for i, (a, b, c, N) in enumerate(settings) if key == f"W[{a},{b}]_c{c}_N{N}"); wlo, whi = settings[s][:2]; R = d[key]
    typed = {a: [[], []] for a in ("RCD", "ICA", "ngBap")}; binacc = {a: [[], []] for a in algos}
    for k in range(R.shape[0]):
        W, pairs = random_instance(np.random.default_rng(40_000 * (s + 1) + k), n, wlo, whi)
        truth = (W.T != 0).astype(int); bi_t = np.zeros((n, n), int)
        for i, j in pairs: bi_t[i, j] = bi_t[j, i] = 1
        for arm in (0, 1):
            e = ae.analyse_estimates(R[k, arm])
            for nm, Bk in (("RCD", e["B_rcd"]), ("ICA", e["B_ica"]), ("ngBap", e["B_ngbap"])):
                bi = np.isnan(Bk) & ~np.eye(n, dtype=bool); bi = bi | bi.T
                typed[nm][arm].append(typed_edge_shd(truth, bi_t, ae.directed_from_B(Bk), bi)["shd"])
                binacc[nm][arm].append(ancestral_binary_accuracy(truth, ae.directed_from_B(Bk)))
            binacc["FCI"][arm].append(ancestral_binary_accuracy(truth, ae.directed_from_pag(e["G_fci"])))
    for a in algos:
        if a in typed: vals.setdefault((a, "typed"), []).append(np.array(typed[a]))
        vals.setdefault((a, "bin"), []).append(np.array(binacc[a]) * 100)
        q = names.index(f"{a} anc AUC"); vals.setdefault((a, "auc"), []).append(np.array([R[:, 0, q], R[:, 1, q]]) * 100)
def cell(a, metric):
    L = vals.get((a, metric))
    if L is None: return ["--"] * len(keys)
    diffs = [(x[1] - x[0]).mean() for x in L]
    ps = [wilcoxon(x[0], x[1]).pvalue if np.any(x[0] != x[1]) else 1.0 for x in L]; adj = holm(np.array(ps))
    return [f"${m:+.1f}$" + ("$^\\star$" if p < 0.05 else "") for m, p in zip(diffs, adj)]
def lab(k):
    w = "0.1--0.5" if "0.1,0.5" in k else "0.5--1.5"; return w, k.split("_c")[1].split("_")[0]
lines = []
for a in algos:
    cs = [cell(a, m) for m in ("typed", "bin", "auc")]
    for i, k in enumerate(keys):
        w, c = lab(k)
        first = (r"\multirow{%d}{*}{%s}" % (len(keys), {"ICA": "ICA-LiNGAM"}.get(a, a))) if i == 0 else ""
        lines.append(f"{first} & {w} & {c} & {cs[0][i]} & {cs[1][i]} & {cs[2][i]} \\\\")
    if a != algos[-1]: lines.append(r"\midrule")
print("\n".join(lines))
