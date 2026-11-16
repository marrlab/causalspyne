"""Rerun of RCD and ICA-LiNGAM (design E, same seeds and data as rcd_dd_vs_sdd.py / ica_dd_vs_sdd.py) that scores the
estimate twice: (old) pair SHD against the ADMG G* given by W and the confounded pairs, and (new) pair SHD against the
exact MAG of the DAG with the hidden roots. MAG: adjacency of the exact PAG (dag2pag), i -> j if i is an ancestor of j,
otherwise i <-> j. Old results are not overwritten: output rcd_ica_mag_E.npz / rcd_ica_mag_E_output.txt.
Columns of the saved array (instance, arm DD/SDD, column): 0 rho(R~), 1 lmin(P), 2-6 RCD metrics vs ADMG (shd, dirR, dirP, biR, biP),
7-11 ICA metrics vs ADMG, 12 RCD pair SHD vs MAG, 13 ICA pair SHD vs MAG.
Usage: python rcd_ica_mag_rescore.py reps [Ns]   (reps=100 and Ns=500,2000 as in the paper; seeds as in rcd_dd_vs_sdd.py E)"""
import sys, os, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import rcd_dd_vs_sdd as r
import fci_dd_vs_pd as fd

def labels_est(B):
    n = B.shape[0]; e = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(B[i, j]) or np.isnan(B[j, i]): e[i, j] = 3
            else: e[i, j] = 1 if B[j, i] != 0 else 2 if B[i, j] != 0 else 0
    return e[np.triu_indices(n, 1)]

def labels_mag(W, T):
    n = W.shape[0]
    reach = (W.T != 0).astype(int)              # reach[i, j] = 1 if i -> j (W[child, parent])
    for _ in range(n): reach = ((reach + reach @ reach) > 0).astype(int)
    t = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            if T[i, j] != 0:
                t[i, j] = 1 if reach[i, j] else 2 if reach[j, i] else 3
    return t[np.triu_indices(n, 1)]

def run_arm(W, pairs, Lam, Ue, UL, A, T):
    from lingam import RCD, ICALiNGAM
    Om = r.omega(Lam)
    X = (Ue + UL @ Lam.T) @ A.T
    X = (X - X.mean(0)) / X.std(0)
    m = RCD(); m.fit(X); B1 = m.adjacency_matrix_
    m2 = ICALiNGAM(random_state=0); m2.fit(X); B2 = m2.adjacency_matrix_
    tm = labels_mag(W, T)
    return (list(fd.spec(W, Om)) + r.rcd_metrics(W, pairs, B1) + r.rcd_metrics(W, pairs, B2)
            + [np.mean(tm != labels_est(B1)), np.mean(tm != labels_est(B2))])

r.run_arm = run_arm

if __name__ == "__main__":
    from scipy.stats import wilcoxon
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    Ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [500, 2000]
    settings = [(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)] for c in [0.7, 2.0] for N in Ns]
    base = 40_000; res = {}
    names = {2: "RCD SHD vs ADMG", 12: "RCD SHD vs MAG", 7: "ICA SHD vs ADMG", 13: "ICA SHD vs MAG"}
    with Pool(int(os.environ.get("NPROC", 16))) as pool:
        for s, (wlo, whi, c, N) in enumerate(settings):
            R = np.array(pool.map(r.one, [("E", base * (s + 1) + k, 10, wlo, whi, c, N, 1500) for k in range(reps)]))
            key = f"W[{wlo},{whi}]_c{c}_N{N}"; res[key] = R
            print(key, "design E,", len(R), "paired instances; means DD, SDD(=RE), difference RE-DD in points, Wilcoxon p", flush=True)
            for q, nm in names.items():
                dd, sd = R[:, 0, q], R[:, 1, q]
                p = wilcoxon(dd, sd).pvalue if np.any(dd != sd) else 1.0
                print(f"   {nm:16s} DD {dd.mean():.3f}  RE {sd.mean():.3f}  diff {100*(sd-dd).mean():+.1f}  p={p:.1e}", flush=True)
            np.savez("rcd_ica_mag_E.npz", **res)
