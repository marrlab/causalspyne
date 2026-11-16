"""Typed-edge SHD (RCD, ICA, ngBap) and ancestral binary accuracy (all four) computed from the stored estimates of ancestral_eval.py
(ancestral_eval_E.npz) without rerunning the algorithms. The truth (W, pairs) is regenerated from the seed of the instance
(seed = 40000 (s + 1) + k, first draw of the generator). Usage: python analyse_stored_estimates.py"""
import numpy as np
from scipy.stats import wilcoxon
import ancestral_eval as ae
from causalspyne.explicit_confounding_gen import random_instance
from causalspyne.pair_shd import typed_edge_shd
from causalspyne.ancestral_acc import ancestral_binary_accuracy
d = np.load("ancestral_eval_E.npz"); n = 10; base = len(ae.column_names())
settings = [(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)] for c in [0.7, 2.0] for N in [500, 2000]]
def holm(p):
    o = np.argsort(p); a = np.empty(len(p)); m = 0
    for r, i in enumerate(o): m = max(m, (len(p) - r) * p[i]); a[i] = min(m, 1)
    return a
rows = {}
for key in d.files:
    s = next(i for i, (a, b, c, N) in enumerate(settings) if key == f"W[{a},{b}]_c{c}_N{N}")
    wlo, whi = settings[s][0], settings[s][1]; R = d[key]; res = {k: [[], []] for k in ("RCD typed", "ICA typed", "ngBap typed", "RCD bin", "ICA bin", "FCI bin", "ngBap bin")}
    for k in range(R.shape[0]):
        W, pairs = random_instance(np.random.default_rng(40_000 * (s + 1) + k), n, wlo, whi)
        truth = (W.T != 0).astype(int); bi_t = np.zeros((n, n), int)
        for i, j in pairs: bi_t[i, j] = bi_t[j, i] = 1
        for arm in (0, 1):
            e = ae.analyse_estimates(R[k, arm])
            for nm, Bk in (("RCD", e["B_rcd"]), ("ICA", e["B_ica"]), ("ngBap", e["B_ngbap"])):
                bi = np.isnan(Bk) & ~np.eye(n, dtype=bool); bi = bi | bi.T
                res[f"{nm} typed"][arm].append(typed_edge_shd(truth, bi_t, ae.directed_from_B(Bk), bi)["shd"] if not np.isnan(Bk).all() else np.nan)
                res[f"{nm} bin"][arm].append(ancestral_binary_accuracy(truth, ae.directed_from_B(Bk)))
            res["FCI bin"][arm].append(ancestral_binary_accuracy(truth, ae.directed_from_pag(e["G_fci"])))
    rows[key] = {k: np.array(v) for k, v in res.items()}
print("CRE - DD (typed SHD in edges, binary ancestral accuracy in points), Wilcoxon p, per setting")
for key, res in rows.items():
    print(key)
    for k, (dd, re_) in res.items():
        ok = ~np.isnan(dd) & ~np.isnan(re_); sc = 100 if "bin" in k else 1
        p = wilcoxon(dd[ok], re_[ok]).pvalue if np.any(dd[ok] != re_[ok]) else 1.0
        print(f"   {k:14s} DD {np.nanmean(dd)*(sc if "bin" in k else 1):.3f} CRE {np.nanmean(re_)*(sc if 'bin' in k else 1):.3f} diff {sc*np.nanmean(re_-dd):+.2f} p={p:.1e}")
