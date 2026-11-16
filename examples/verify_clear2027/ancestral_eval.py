"""RCD, ICA-LiNGAM, FCI and ngBap on the data of rcd_dd_vs_sdd.py (design E: same seeds, instances and data; DD vs root-explicit RE),
scored only with causalspyne functions: causalspyne.ancestral_acc (ancestral pair accuracy / precision / recall, ancestral AUC,
total-effect and PAG ancestor scores) and causalspyne.pair_shd (pair SHD against the ADMG and the exact MAG). The estimated graphs
(RCD, ICA, ngBap: B matrix, FCI: causal-learn PAG matrix) are stored in the npz, so every further metric can be computed without
rerunning the algorithms (see analyse_estimates() below).
Usage: python ancestral_eval.py reps [Ns]    (NPROC env var; default Ns = 2000; ngBap needs Rscript with the package in ~/Rlibs, see ngbap_run.R)
Saved array R[instance, arm (0 DD, 1 RE), column]: columns 0-1 rho(R~), lmin(P); then for each algorithm in ALGOS the metrics in
METRICS (acc, prec, rec, auc); then pair SHD of ngBap vs ADMG, vs MAG; then the 4 estimates flattened (n*n each)."""
import sys, os, subprocess, tempfile, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import rcd_dd_vs_sdd as r
import fci_dd_vs_pd as fd
from causalspyne.ancestral_acc import (ancestral_relation_scores, ancestral_auc, total_effect_ancestor_scores,
                                       pag_ancestor_scores, ancestor_closure)
from causalspyne.pair_shd import labels_admg, labels_estimate, labels_exact_mag, pair_shd

ALGOS = ["RCD", "ICA", "FCI", "ngBap"]
METRICS = ["anc acc", "anc prec", "anc rec", "anc AUC"]
N_NODES = 10
HERE = os.path.dirname(os.path.abspath(__file__))

def directed_from_B(B):
    D = (np.nan_to_num(B) != 0).astype(int).T
    np.fill_diagonal(D, 0); return D

def directed_from_pag(G):
    n = G.shape[0]; D = np.zeros((n, n), int)
    for i in range(n):
        for j in range(n):
            if i != j and G[j, i] == 1 and G[i, j] == -1: D[i, j] = 1
    return D

def run_ngbap(X):
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(os.path.join(d, "X.csv"), X, delimiter=",")
        subprocess.run(["Rscript", os.path.join(HERE, "ngbap_run.R"), os.path.join(d, "X.csv"), os.path.join(d, "o"), "4"],
                       check=True, capture_output=True, timeout=1800, env=dict(os.environ, OMP_NUM_THREADS="1"))
        B = np.loadtxt(os.path.join(d, "o_B.csv"), delimiter=","); bi = np.loadtxt(os.path.join(d, "o_bi.csv"), delimiter=",")
    B = B.copy(); B[(bi != 0) & ~np.eye(len(B), dtype=bool)] = np.nan
    return B

def metrics_from_estimates(truth, B1, B2, G, B3):
    out = []
    for kind, est in (("B", B1), ("B", B2), ("G", G), ("B", B3)):
        if est is None or np.isnan(est).all():
            out += [np.nan] * 4; continue
        if kind == "B":
            acc = ancestral_relation_scores(truth, directed_from_B(est)); auc = ancestral_auc(truth, total_effect_ancestor_scores(est))
        else:
            acc = ancestral_relation_scores(truth, directed_from_pag(est)); auc = ancestral_auc(truth, pag_ancestor_scores(est))
        out += list(acc) + [auc]
    return out

def run_arm(W, pairs, Lam, Ue, UL, A, T):
    from lingam import RCD, ICALiNGAM
    from causallearn.search.ConstraintBased.FCI import fci
    n = W.shape[0]; Om = r.omega(Lam)
    X = (Ue + UL @ Lam.T) @ A.T; X = (X - X.mean(0)) / X.std(0)
    truth = (W.T != 0).astype(int)
    m = RCD(); m.fit(X); B1 = np.array(m.adjacency_matrix_, float)
    m2 = ICALiNGAM(random_state=0); m2.fit(X); B2 = np.array(m2.adjacency_matrix_, float)
    G = np.array(fci(X, "fisherz", 0.05, verbose=False, show_progress=False)[0].graph, float)
    try:
        B3 = run_ngbap(X)
        shd = [pair_shd(labels_admg(W, pairs), labels_estimate(B3)), pair_shd(labels_exact_mag(W, pairs, pag=T), labels_estimate(B3))]
    except Exception:
        B3 = np.full((n, n), np.nan); shd = [np.nan, np.nan]
    return np.array(list(fd.spec(W, Om)) + metrics_from_estimates(truth, B1, B2, G, B3) + shd
                    + list(B1.ravel()) + list(B2.ravel()) + list(G.ravel()) + list(B3.ravel()))

r.run_arm = run_arm

def column_names():
    return ["rho(R~)", "lmin(P)"] + [f"{a} {m}" for a in ALGOS for m in METRICS] + ["ngBap SHD vs ADMG", "ngBap SHD vs MAG"]

def analyse_estimates(R, W_list=None):
    """Hook for metrics computed later from the stored estimates: R[..., 18:] holds B_rcd, B_ica, G (PAG), B_ngbap flattened."""
    n2 = N_NODES * N_NODES
    base = len(column_names())
    return {k: R[..., base + i * n2: base + (i + 1) * n2].reshape(R.shape[:-1] + (N_NODES, N_NODES)) for i, k in enumerate(["B_rcd", "B_ica", "G_fci", "B_ngbap"])}

if __name__ == "__main__":
    from scipy.stats import wilcoxon
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    Ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2000]
    settings = [(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)] for c in [0.7, 2.0] for N in [500, 2000]]
    names = column_names(); res = {}
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        for s, (wlo, whi, c, N) in enumerate(settings):
            if N not in Ns: continue
            R = np.array(pool.map(r.one, [("E", 40_000 * (s + 1) + k, 10, wlo, whi, c, N, 1500) for k in range(reps)]))
            key = f"W[{wlo},{whi}]_c{c}_N{N}"; res[key] = R; np.savez("ancestral_eval_E.npz", **res)
            print(key, f"design E, {len(R)} paired instances; means DD, RE, RE-DD in points, Wilcoxon p", flush=True)
            for q in range(2, len(names)):
                dd, sd = R[:, 0, q], R[:, 1, q]; ok = ~np.isnan(dd) & ~np.isnan(sd)
                p = wilcoxon(dd[ok], sd[ok]).pvalue if ok.any() and np.any(dd[ok] != sd[ok]) else 1.0
                print(f"   {names[q]:18s} DD {np.nanmean(dd):.3f}  RE {np.nanmean(sd):.3f}  diff {100*np.nanmean(sd-dd):+.1f}  p={p:.1e}  (n={ok.sum()})", flush=True)
