"""Does the Omega class change the ranking of FCI and RCD?  Non-Gaussian (uniform) data, same ADMG, design E extremes.

One common metric for both algorithms, per unordered pair of observed variables the type is one of
  0 none, 1 i->j, 2 j->i, 3 i<->j.   Truth: bidirected pair -> 3, else the directed edge, else 0.
  pair error     : share of pairs whose type differs from the truth (FCI output with a circle mark counts as wrong)
  adjacency error: share of pairs whose adjacency (type != 0) differs from the truth
Per ADMG and arm (DD, RE = root-explicit with unequal loadings): error of FCI, error of RCD; the 'winner' is the
algorithm with the smaller error (ties split).  We report the share of ADMGs won by RCD per arm, the mean of
(error_FCI - error_RCD) per arm, and a Wilcoxon test of whether this difference changes between the arms.
Usage: python rank_fci_rcd.py [instances]"""
import sys, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import rcd_dd_vs_sdd as rd, fci_rejection as fr, fci_dd_vs_pd as fd

def fci_types(G):
    n = G.shape[0]; out = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = G[i, j], G[j, i]           # a: mark at i, b: mark at j
            if a == 0 and b == 0: t = 0
            elif a == -1 and b == 1: t = 1     # i -> j
            elif a == 1 and b == -1: t = 2
            elif a == 1 and b == 1: t = 3
            else: t = 9                        # circle / undirected: undetermined
            out.append(t)
    return np.array(out)

def rcd_types(B):
    n = B.shape[0]; out = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(B[i, j]) or np.isnan(B[j, i]): t = 3
            else: t = 1 if B[j, i] != 0 else 2 if B[i, j] != 0 else 0
            out.append(t)
    return np.array(out)

def truth_types(W, pairs):
    n = W.shape[0]; conf = set(map(tuple, pairs)) | {(j, i) for i, j in pairs}; out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(3 if (i, j) in conf else 1 if W[j, i] else 2 if W[i, j] else 0)
    return np.array(out)

def one(a):
    s, r, N = a
    from lingam import RCD
    from causallearn.search.ConstraintBased.FCI import fci
    wlo, whi, c = [(0.1, 0.5, 0.7), (0.1, 0.5, 2.0), (0.5, 1.5, 0.7), (0.5, 1.5, 2.0)][s]
    n = 10
    rng = np.random.default_rng(40_000 * (s + 1) + r)
    W, pairs = fr.instance(rng, n, wlo, whi)
    k = len(pairs)
    Lams = [rd.search(rng, W, pairs, n, cls, c, 1500) for cls in ("DD", "SDD")]
    A = np.linalg.inv(np.eye(n) - W)
    s3 = np.sqrt(3)
    Ue = rng.uniform(-s3, s3, (N, n)); UL = rng.uniform(-s3, s3, (N, k))
    T = truth_types(W, pairs); out = np.zeros((2, 4))
    for a_, L in enumerate(Lams):
        X = (Ue + UL @ L.T) @ A.T; X = (X - X.mean(0)) / X.std(0)
        m = RCD(); m.fit(X)
        G, _ = fci(X, "fisherz", 0.05, verbose=False, show_progress=False)
        f, q = fci_types(G.graph), rcd_types(m.adjacency_matrix_)
        out[a_] = [np.mean(f != T), np.mean(q != T), np.mean((f != 0) != (T != 0)), np.mean((q != 0) != (T != 0))]
    return out

if __name__ == "__main__":
    from scipy.stats import wilcoxon
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    lines, store = [], {}
    with Pool(16) as pool:
        for s, (wlo, whi, c) in enumerate([(0.1, 0.5, 0.7), (0.1, 0.5, 2.0), (0.5, 1.5, 0.7), (0.5, 1.5, 2.0)]):
            for N in (500, 2000):
                M = np.array(pool.map(one, [(s, r, N) for r in range(R)]))  # R x arm x 4
                store[f"s{s}_N{N}"] = M
                lines.append(f"W[{wlo},{whi}] c={c} N={N} ({R} ADMGs)")
                for nm, (fi, ri) in (("pair error", (0, 1)), ("adjacency error", (2, 3))):
                    d = M[:, :, fi] - M[:, :, ri]            # FCI - RCD per arm; >0: RCD better
                    win = [(np.mean(d[:, a] > 0) + 0.5 * np.mean(d[:, a] == 0)) for a in (0, 1)]
                    p = wilcoxon(d[:, 1] - d[:, 0]).pvalue if np.any(d[:, 1] != d[:, 0]) else float("nan")
                    lines.append(f"   {nm:16s} FCI/RCD under DD {M[:,0,fi].mean():.3f}/{M[:,0,ri].mean():.3f}  RE {M[:,1,fi].mean():.3f}/{M[:,1,ri].mean():.3f}"
                                 f"   RCD wins {win[0]:.2f} (DD) -> {win[1]:.2f} (RE)   mean(FCI-RCD) {d[:,0].mean():+.3f} -> {d[:,1].mean():+.3f}  p(change)={p:.1e}")
                print("\n".join(lines[-3:]), flush=True)
    np.savez("rank_fci_rcd.npz", **store)
    open("rank_fci_rcd_output.txt", "w").write("\n".join(lines) + "\n")
