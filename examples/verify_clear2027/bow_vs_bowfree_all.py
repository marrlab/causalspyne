"""Bow-free vs bow version of the same instance, for FCI, RCD, ICA-LiNGAM and ngBap (uniform noise).
Arm A (bow-free): W, confounded pairs (no directed edge between the two vertices of a pair), Omega = I + Lambda Lambda^T with one root
latent per pair (diagonally dominant when the loadings are equal), as in rcd_dd_vs_sdd.py. Arm B (bows): the same, plus a directed edge
(oriented along a topological order, weight in the same range) for a random half of the pairs, so these pairs are bows. Same W otherwise,
same pairs, same Omega, same noise draws.
Metrics (all from causalspyne): typed_edge_shd (RCD, ICA, ngBap; bows count as -> and <->), ancestral_binary_accuracy and ancestral_auc
(all four), pair-free PAG SHD of FCI against the exact PAG of the DAG with one hidden root per pair (valid with bows).
Saved: bow_all_E.npz, R[instance, arm (0 bow-free, 1 bow), column]; columns in column_names(); estimates stored after the metrics.
Usage: python bow_vs_bowfree_all.py reps    (NPROC env var; settings: |W| in [0.1,0.5], [0.5,1.5]; c = 2; N = 2000)"""
import sys, os, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import fci_rejection as fr
import fci_dd_vs_pd as fd
import rcd_dd_vs_sdd as r
import ancestral_eval as ae
from causalspyne.ancestral_acc import ancestral_binary_accuracy, ancestral_auc, total_effect_ancestor_scores, pag_ancestor_scores
from causalspyne.pair_shd import typed_edge_shd, exact_pag

def column_names():
    c = []
    for a in ("RCD", "ICA", "ngBap"):
        c += [f"{a} typed SHD", f"{a} dir errors", f"{a} bi errors"]
    c += [f"{a} anc binary acc" for a in ae.ALGOS] + [f"{a} anc AUC" for a in ae.ALGOS] + ["FCI PAG SHD"]
    return c

def add_bows(rng, W, pairs, wlo, whi):
    n = len(W); reach = ae.ancestor_closure((W.T != 0).astype(int))
    pos = np.argsort(np.argsort(reach.sum(0), kind="stable"), kind="stable")   # reach.sum(0)[v] = number of ancestors of v
    chosen = rng.choice(len(pairs), size=max(1, len(pairs) // 2), replace=False)
    W2 = W.copy()
    for l in chosen:
        i, j = pairs[l]; a, b = (i, j) if pos[i] < pos[j] else (j, i)          # a -> b
        W2[b, a] = rng.uniform(wlo, whi) * rng.choice([-1, 1])
    return W2

def arm(W, pairs, Lam, Ue, UL):
    from lingam import RCD, ICALiNGAM
    from causallearn.search.ConstraintBased.FCI import fci
    n = len(W); A = np.linalg.inv(np.eye(n) - W)
    X = (Ue + UL @ Lam.T) @ A.T; X = (X - X.mean(0)) / X.std(0)
    truth = (W.T != 0).astype(int)
    bi_true = np.zeros((n, n), int)
    for i, j in pairs: bi_true[i, j] = bi_true[j, i] = 1
    m = RCD(); m.fit(X); B1 = np.array(m.adjacency_matrix_, float)
    m2 = ICALiNGAM(random_state=0); m2.fit(X); B2 = np.array(m2.adjacency_matrix_, float)
    T = exact_pag(W, pairs)
    G = np.array(fci(X, "fisherz", 0.05, verbose=False, show_progress=False)[0].graph, float)
    try: B3 = ae.run_ngbap(X)
    except Exception: B3 = np.full((n, n), np.nan)
    out, ests = [], []
    for B in (B1, B2, B3):
        if np.isnan(B).all(): out += [np.nan] * 3; continue
        bi_est = np.isnan(B) & ~np.eye(n, dtype=bool); bi_est = bi_est | bi_est.T
        res = typed_edge_shd(truth, bi_true, ae.directed_from_B(B), bi_est)
        out += [res["shd"], sum(res["dir"].values()), sum(res["bi"].values())]
    Ds = [ae.directed_from_B(B1), ae.directed_from_B(B2), ae.directed_from_pag(G), ae.directed_from_B(B3)]
    out += [ancestral_binary_accuracy(truth, D) if not np.isnan(B3).all() or k != 3 else np.nan for k, D in enumerate(Ds)]
    scores = [total_effect_ancestor_scores(B1), total_effect_ancestor_scores(B2), pag_ancestor_scores(G), total_effect_ancestor_scores(B3)]
    out += [ancestral_auc(truth, S) if not np.isnan(B3).all() or k != 3 else np.nan for k, S in enumerate(scores)]
    out += [fd.metrics(T, G)[0]]
    return out + list(B1.ravel()) + list(B2.ravel()) + list(G.ravel()) + list(B3.ravel())

def one(args):
    seed, n, wlo, whi, c, N = args
    rng = np.random.default_rng(seed)
    W, pairs = fr.instance(rng, n, wlo, whi); k = len(pairs)
    W2 = add_bows(rng, W, pairs, wlo, whi)
    th = rng.uniform(-c, c, k); Lam = r.loadings(th, np.zeros(k), pairs, n)
    s3 = np.sqrt(3); Ue = rng.uniform(-s3, s3, (N, n)); UL = rng.uniform(-s3, s3, (N, k))
    return np.array([arm(W, pairs, Lam, Ue, UL), arm(W2, pairs, Lam, Ue, UL)])

if __name__ == "__main__":
    from scipy.stats import wilcoxon
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    names = column_names(); res = {}
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        for s, (wlo, whi) in enumerate([(0.1, 0.5), (0.5, 1.5)]):
            R = np.array(pool.map(one, [(90_000 * (s + 1) + k, 10, wlo, whi, 2.0, 2000) for k in range(reps)]))
            key = f"W[{wlo},{whi}]_c2.0_N2000"; res[key] = R; np.savez("bow_all_E.npz", **res)
            print(key, f"{len(R)} paired instances; means bow-free (A), bows (B), B-A, Wilcoxon p", flush=True)
            for q, nm in enumerate(names):
                a, b = R[:, 0, q], R[:, 1, q]; ok = ~np.isnan(a) & ~np.isnan(b)
                p = wilcoxon(a[ok], b[ok]).pvalue if ok.any() and np.any(a[ok] != b[ok]) else 1.0
                print(f"   {nm:22s} A {np.nanmean(a):.3f}  B {np.nanmean(b):.3f}  diff {np.nanmean(b-a):+.3f}  p={p:.1e}  (n={ok.sum()})", flush=True)
