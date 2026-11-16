"""FCI with implicit sampling: bow-free vs bow version of the same instance.
X = (I - W)^{-1} eps, eps ~ N(0, Omega), Omega = I + diagonal-dominant theta-pattern (diag_i = 1 + sum_j |theta_ij|, off-diag theta_ij):
the paper's DD construction. Arm A (bow-free): confounded pairs between vertices with no directed edge (as the generator of the other
experiments). Arm B (bows): the same number of pairs, chosen among pairs joined by a directed edge (bows). Same W, same theta values,
same standard-normal draws. Truth = exact PAG of the latent projection (hidden root per pair, dag2pag; valid with a bow).
Metrics: normalized SHD of FCI (fisherz, 0.05) to the PAG, adjacency precision and recall, number of extra adjacencies of the PAG vs the ADMG.
Usage: python fci_bow_vs_bowfree.py reps   (NPROC env var = number of processes). Writes fci_bow_E.npz and prints means / Wilcoxon."""
import sys, os, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import fci_rejection as fr
import fci_dd_vs_pd as fd

def omega_dd(th, pairs, n):
    Om = np.zeros((n, n))
    for t, (i, j) in zip(th, pairs):
        Om[i, j] = Om[j, i] = t; Om[i, i] += abs(t); Om[j, j] += abs(t)
    return Om + np.eye(n)

def bow_pairs(rng, W, k, n):
    cand = [(i, j) for i in range(n) for j in range(i + 1, n) if W[i, j] != 0 or W[j, i] != 0]
    rng.shuffle(cand); out, deg = [], np.zeros(n, int)
    for i, j in cand:
        if len(out) == k: break
        if deg[i] < 2 and deg[j] < 2: out.append((i, j)); deg[i] += 1; deg[j] += 1
    return out

def one(args):
    seed, n, wlo, whi, c, N = args
    from causallearn.search.ConstraintBased.FCI import fci
    rng = np.random.default_rng(seed)
    W, pairsA = fr.instance(rng, n, wlo, whi)
    pairsB = bow_pairs(rng, W, len(pairsA), n)
    k = len(pairsB)
    if k == 0: return np.full((2, 5), np.nan)
    pairsA = pairsA[:k]
    th = rng.uniform(-c, c, k); Z = rng.standard_normal((N, n)); A = np.linalg.inv(np.eye(n) - W)
    out = []
    for pairs in (pairsA, pairsB):
        Om = omega_dd(th, pairs, n)
        X = (Z @ np.linalg.cholesky(Om).T) @ A.T
        T = fd.true_pag(W, pairs)
        G, _ = fci(X, "fisherz", 0.05, verbose=False, show_progress=False)
        m = fd.metrics(T, G.graph)
        adjA = (W != 0) | (W.T != 0)
        for i, j in pairs: adjA[i, j] = adjA[j, i] = True
        extra = int((((T != 0) & ~np.eye(n, dtype=bool)) & ~adjA).sum() // 2)
        out.append([m[0], m[1], m[2], extra, k])
    return np.array(out)

if __name__ == "__main__":
    from scipy.stats import wilcoxon
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    settings = [(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)] for c in [0.7, 2.0] for N in [500, 2000]]
    res = {}
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        for s, (wlo, whi, c, N) in enumerate(settings):
            R = np.array(pool.map(one, [(70_000 * (s + 1) + r, 10, wlo, whi, c, N) for r in range(reps)]))
            R = R[~np.isnan(R[:, 0, 0])]
            key = f"W[{wlo},{whi}]_c{c}_N{N}"; res[key] = R
            print(key, f"design implicit DD, {len(R)} paired instances; means bow-free (A), bows (B), B-A in points, Wilcoxon p", flush=True)
            for q, nm in [(0, "FCI SHD"), (1, "adj precision"), (2, "adj recall"), (3, "extra PAG adjacencies")]:
                a, b = R[:, 0, q], R[:, 1, q]
                p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
                print(f"   {nm:22s} A {a.mean():.3f}  B {b.mean():.3f}  diff {100*(b-a).mean() if q<3 else (b-a).mean():+.1f}  p={p:.1e}", flush=True)
            np.savez("fci_bow_E.npz", **res)
