"""FCI on the HARDEST reachable instances of each Omega class, same fixed W and pattern.
Per instance: search (fixedW_extremes.search) for the Omega minimizing lambda_min(P) within
DD and within PD (same pattern, |theta|<=c, lambda_min(Omega)>=1), then run FCI on both."""
import sys, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import fixedW_extremes as fx
import fci_dd_vs_pd as fd

def argsearch(rng, W, pairs, n, cls, c, budget):
    best, bestv = None, -np.inf
    k = len(pairs)
    f = lambda th, dv: -fx.stats(W, fx.build(th, dv, pairs, n, cls))[1]
    for _ in range(budget // 4):
        th = rng.uniform(-c, c, k); dv = rng.uniform(0, 6 * c, n)
        v = f(th, dv)
        if v > bestv: best, bestv = (th, dv), v
    th, dv = best[0].copy(), best[1].copy(); step = c / 2
    for it in range(budget):
        th2, dv2 = th.copy(), dv.copy()
        if k and rng.random() < 0.7:
            m = rng.integers(k); th2[m] = np.clip(th2[m] + rng.normal(0, step), -c, c)
        else:
            m = rng.integers(n); dv2[m] = max(0.0, dv2[m] + rng.normal(0, step))
        v = f(th2, dv2)
        if v > bestv: th, dv, bestv = th2, dv2, v
        if it % 300 == 299: step *= 0.7
    return fx.build(th, dv, pairs, n, cls)

def one(args):
    seed, n, wlo, whi, c, N, alpha, budget = args
    from causallearn.search.ConstraintBased.FCI import fci
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n)); m = np.tril(rng.random((n, n)) < 0.3, -1)
    W[m] = rng.uniform(wlo, whi, m.sum()) * rng.choice([-1, 1], m.sum())
    perm = rng.permutation(n); W = W[np.ix_(perm, perm)]
    pairs, deg = [], np.zeros(n, int)
    for _ in range(4 * n):
        i, j = sorted(rng.choice(n, 2, replace=False))
        if deg[i] < 2 and deg[j] < 2 and (i, j) not in pairs and W[i, j] == 0 and W[j, i] == 0:
            pairs.append((i, j)); deg[i] += 1; deg[j] += 1
    T = fd.true_pag(W, pairs)
    A = np.linalg.inv(np.eye(n) - W)
    Z = rng.standard_normal((N, n))
    out = []
    for cls in ("DD", "PD"):
        Om = argsearch(rng, W, pairs, n, cls, c, budget)
        X = Z @ np.linalg.cholesky(Om).T @ A.T
        X = (X - X.mean(0)) / X.std(0)
        G, _ = fci(X, "fisherz", alpha, verbose=False, show_progress=False)
        out.append(fd.spec(W, Om) + fd.metrics(T, G.graph))
    return np.array(out)

if __name__ == "__main__":
    from scipy.stats import wilcoxon
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    names = ["rho(R~)", "lmin(P)", "SHD", "adj P", "adj R", "arrow P", "arrow R"]
    res = {}
    with Pool(16) as pool:
        for s, (wlo, whi, c, N) in enumerate([(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)]
                                               for c in [0.7, 2.0] for N in [500, 5000]]):
            R = np.array(pool.map(one, [(20_000 * s + r, 10, wlo, whi, c, N, 0.05, 1500) for r in range(reps)]))
            key = f"W[{wlo},{whi}]_c{c}_N{N}"; res[key] = R
            print(f"{key}  ({reps} paired instances, means; Wilcoxon p)")
            for k, nm in enumerate(names):
                dd, pd = R[:, 0, k], R[:, 1, k]
                p = wilcoxon(dd, pd).pvalue if np.any(dd != pd) else 1.0
                print(f"   {nm:8s} DD {dd.mean():.3f}  PD {pd.mean():.3f}  p={p:.1e}", flush=True)
    np.savez("fci_extremes.npz", **res)
