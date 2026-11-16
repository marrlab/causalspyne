"""ngBap (non-Gaussian bow-free acyclic path diagrams, Wang and Drton; R package, K = 4 because the noise is uniform) together with
RCD, ICA-LiNGAM and FCI on the same data as rcd_dd_vs_sdd.py (design E, same seeds and instances, DD vs root-explicit RE), scored by
(i) ancestral accuracy (causalspyne.ancestral_acc.ancestral_relation_scores: pair accuracy, precision and recall of 'i is an ancestor of j',
estimated ancestors = transitive closure of the definite directed edges) for all four algorithms, and (ii) the pair SHD of ngBap against
the ADMG and against the exact MAG (as in rcd_ica_mag_rescore.py). Old results are not touched; writes ngbap_ancestral_E.npz.
Usage: python ngbap_ancestral.py reps [Ns]    (NPROC env var; default Ns = 2000; needs Rscript with ngBap in ~/Rlibs)
Columns (instance, arm DD/RE, column): 0 rho(R~), 1 lmin(P); then for RCD, ICA, FCI, ngBap: anc pair acc, anc precision, anc recall
(2-4, 5-7, 8-10, 11-13); 14 ngBap pair SHD vs ADMG, 15 ngBap pair SHD vs MAG."""
import sys, os, subprocess, tempfile, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import rcd_dd_vs_sdd as r
import fci_dd_vs_pd as fd
import rcd_ica_mag_rescore as mg
from causalspyne.ancestral_acc import ancestral_relation_scores

HERE = os.path.dirname(os.path.abspath(__file__))

def directed_from_B(B):          # B[i, j] != 0 (not NaN): j -> i  ->  D[j, i] = 1
    D = (np.nan_to_num(B) != 0).astype(int).T
    np.fill_diagonal(D, 0); return D

def directed_from_pag(G):        # causal-learn: G[j, i] == 1 and G[i, j] == -1  <=>  i -> j
    n = G.shape[0]; D = np.zeros((n, n), int)
    for i in range(n):
        for j in range(n):
            if i != j and G[j, i] == 1 and G[i, j] == -1: D[i, j] = 1
    return D

def run_ngbap(X):
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(os.path.join(d, "X.csv"), X, delimiter=",")
        env = dict(os.environ, OMP_NUM_THREADS="1")
        subprocess.run(["Rscript", os.path.join(HERE, "ngbap_run.R"), os.path.join(d, "X.csv"), os.path.join(d, "o"), "4"],
                       check=True, capture_output=True, timeout=1800, env=env)
        B = np.loadtxt(os.path.join(d, "o_B.csv"), delimiter=","); bi = np.loadtxt(os.path.join(d, "o_bi.csv"), delimiter=",")
    B = B.copy(); conf = (bi != 0) & ~np.eye(len(B), dtype=bool)
    B[conf] = np.nan                                        # confounded pair: NaN, as in RCD's output
    return B

def run_arm(W, pairs, Lam, Ue, UL, A, T):
    from lingam import RCD, ICALiNGAM
    from causallearn.search.ConstraintBased.FCI import fci
    Om = r.omega(Lam); n = W.shape[0]
    X = (Ue + UL @ Lam.T) @ A.T; X = (X - X.mean(0)) / X.std(0)
    truth = (W.T != 0).astype(int)                         # truth[i, j] = 1 if i -> j
    m = RCD(); m.fit(X); B1 = m.adjacency_matrix_
    m2 = ICALiNGAM(random_state=0); m2.fit(X); B2 = m2.adjacency_matrix_
    G, _ = fci(X, "fisherz", 0.05, verbose=False, show_progress=False)
    try:
        B3 = run_ngbap(X)
        ng = list(ancestral_relation_scores(truth, directed_from_B(B3)))
        shd_admg = r.rcd_metrics(W, pairs, B3)[0]; shd_mag = float(np.mean(mg.labels_mag(W, T) != mg.labels_est(B3)))
    except Exception as e:                                  # keep going, count as missing
        ng = [np.nan] * 3; shd_admg = shd_mag = np.nan
    return (list(fd.spec(W, Om)) + list(ancestral_relation_scores(truth, directed_from_B(B1)))
            + list(ancestral_relation_scores(truth, directed_from_B(B2)))
            + list(ancestral_relation_scores(truth, directed_from_pag(G.graph))) + ng + [shd_admg, shd_mag])

r.run_arm = run_arm

if __name__ == "__main__":
    from scipy.stats import wilcoxon
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    Ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2000]
    settings = [(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)] for c in [0.7, 2.0] for N in [500, 2000]]
    names = {2: "RCD anc acc", 3: "RCD anc prec", 4: "RCD anc rec", 5: "ICA anc acc", 6: "ICA anc prec", 7: "ICA anc rec",
             8: "FCI anc acc", 9: "FCI anc prec", 10: "FCI anc rec", 11: "ngBap anc acc", 12: "ngBap anc prec", 13: "ngBap anc rec",
             14: "ngBap SHD vs ADMG", 15: "ngBap SHD vs MAG"}
    res = {}
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        for s, (wlo, whi, c, N) in enumerate(settings):
            if N not in Ns: continue
            R = np.array(pool.map(r.one, [("E", 40_000 * (s + 1) + k, 10, wlo, whi, c, N, 1500) for k in range(reps)]))
            key = f"W[{wlo},{whi}]_c{c}_N{N}"; res[key] = R; np.savez("ngbap_ancestral_E.npz", **res)
            print(key, f"design E, {len(R)} paired instances; means DD, RE, difference RE-DD in points, Wilcoxon p", flush=True)
            for q, nm in names.items():
                dd, sd = R[:, 0, q], R[:, 1, q]; ok = ~np.isnan(dd) & ~np.isnan(sd)
                p = wilcoxon(dd[ok], sd[ok]).pvalue if np.any(dd[ok] != sd[ok]) else 1.0
                print(f"   {nm:18s} DD {np.nanmean(dd):.3f}  RE {np.nanmean(sd):.3f}  diff {100*np.nanmean(sd-dd):+.1f}  p={p:.1e}  (n={ok.sum()})", flush=True)
