"""User's design (rejection threshold = 95th pct of random DD draws; DD max is reported but random PD draws essentially never exceed it): for a FIXED W and pattern, keep only PD (explicit-reachable) Omegas whose
rho(R~) exceeds the maximum reachable by ANY diagonally dominant Omega (same pattern, |theta|<=c,
lambda_min(Omega)>=1). Compare FCI on such PD draws vs ordinary random DD draws."""
import sys, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import fixedW_extremes as fx
import fci_dd_vs_pd as fd

def instance(rng, n, wlo, whi):
    W = np.zeros((n, n)); m = np.tril(rng.random((n, n)) < 0.3, -1)
    W[m] = rng.uniform(wlo, whi, m.sum()) * rng.choice([-1, 1], m.sum())
    perm = rng.permutation(n); W = W[np.ix_(perm, perm)]
    pairs, deg = [], np.zeros(n, int)
    for _ in range(4 * n):
        i, j = sorted(rng.choice(n, 2, replace=False))
        if deg[i] < 2 and deg[j] < 2 and (i, j) not in pairs and W[i, j] == 0 and W[j, i] == 0:
            pairs.append((i, j)); deg[i] += 1; deg[j] += 1
    return W, pairs

def one(args):
    seed, n, wlo, whi, c, N, alpha, tries = args
    from causallearn.search.ConstraintBased.FCI import fci
    rng = np.random.default_rng(seed)
    W, pairs = instance(rng, n, wlo, whi)
    k = len(pairs)
    dd_max = fx.search(rng, W, pairs, n, "DD", "rho", c, 1500)
    dd_rand = [fd.spec(W, fx.build(rng.uniform(-c, c, k), None, pairs, n, "DD"))[0] for _ in range(2000)]
    thr = np.quantile(dd_rand, 0.95)
    pd_om = None
    for t in range(tries):  # rejection sampling in the PD class
        th = rng.uniform(-c, c, k) * (rng.random(k) < 1)  # full pattern
        dv = rng.exponential(0.3, n)
        Om = fx.build(th, dv, pairs, n, "PD")
        if fd.spec(W, Om)[0] > thr:
            pd_om = Om; break
    if pd_om is None:
        return None
    th = rng.uniform(-c, c, k)
    dd_om = fx.build(th, None, pairs, n, "DD")  # ordinary random DD draw
    T = fd.true_pag(W, pairs)
    A = np.linalg.inv(np.eye(n) - W)
    Z = rng.standard_normal((N, n))
    out = []
    for Om in (dd_om, pd_om):
        X = Z @ np.linalg.cholesky(Om).T @ A.T
        X = (X - X.mean(0)) / X.std(0)
        G, _ = fci(X, "fisherz", alpha, verbose=False, show_progress=False)
        out.append(fd.spec(W, Om) + fd.metrics(T, G.graph))
    return np.array(out), (dd_max, thr), t + 1

if __name__ == "__main__":
    from scipy.stats import wilcoxon
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    names = ["rho(R~)", "lmin(P)", "SHD", "adj P", "adj R", "arrow P", "arrow R"]
    res = {}
    with Pool(16) as pool:
        for s, (wlo, whi, c, N) in enumerate([(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)]
                                               for c in [0.7, 2.0] for N in [500, 5000]]):
            outs = pool.map(one, [(30_000 * s + r, 10, wlo, whi, c, N, 0.05, 20000) for r in range(reps)])
            ok = [o for o in outs if o is not None]
            R = np.array([o[0] for o in ok]); ddm = np.array([o[1] for o in ok]); tr = np.array([o[2] for o in ok])
            key = f"W[{wlo},{whi}]_c{c}_N{N}"; res[key] = R
            print(f"{key}: accepted {len(ok)}/{reps} W (median tries {np.median(tr):.0f}); median rho: DD max {np.median(ddm[:,0]):.2f}, DD q95 threshold {np.median(ddm[:,1]):.2f}")
            for kk, nm in enumerate(names):
                dd, pd = R[:, 0, kk], R[:, 1, kk]
                p = wilcoxon(dd, pd).pvalue if np.any(dd != pd) else 1.0
                print(f"   {nm:8s} DD {dd.mean():.3f}  PD>q95 {pd.mean():.3f}  p={p:.1e}", flush=True)
    np.savez("fci_rejection.npz", **res)
