"""RCD (repetitive causal discovery, lingam package; handles hidden confounders, needs non-Gaussian
data) under diagonally dominant vs root-explicit (unequal loadings) confounding, same fixed ADMG.

Both classes are explicit latent models with one root latent per bidirected pair:
  X = (I - W)^{-1} (e + sum_l lam_l L_l),  lam_l = a_l e_i + s_l b_l e_j,  a_l b_l = |theta_l|,
  e_i, L_l iid uniform with unit variance (non-Gaussian), so Omega = I + sum lam_l lam_l^T.
  DD : a_l = b_l (the paper's construction).   SDD : a_l / b_l = e^{2 r_l}, |r_l| <= 2.
Design E (extremes): per class, the (theta, r) minimizing lambda_min(P) (search of F.5).
Design R (random): theta ~ U(-c, c); DD vs SDD with r ~ U(-2, 2), same theta.
Within a pair: same W, pattern, theta (design R), and the same uniform draws.
Metrics (RCD output B: B[i,j] != 0 means j -> i, NaN means confounded pair), over the n(n-1)/2
unordered pairs with truth in {none, i->j, j->i, <->}: normalized pair-SHD, directed-edge recall
and precision (correct orientation), bidirected recall and precision. FCI (fisherz) normalized SHD
to the true PAG on the same data for reference.
Usage: python rcd_dd_vs_sdd.py E|R reps
"""
import sys, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import fixedW_extremes as fx
import fci_dd_vs_pd as fd
import fci_rejection as fr


def loadings(th, r, pairs, n):
    Lam = np.zeros((n, len(pairs)))
    for l, ((i, j), t, rr) in enumerate(zip(pairs, th, r)):
        a, b = np.sqrt(abs(t)) * np.exp(rr), np.sqrt(abs(t)) * np.exp(-rr)
        Lam[i, l], Lam[j, l] = a, np.sign(t) * b
    return Lam


def omega(Lam):
    return np.eye(Lam.shape[0]) + Lam @ Lam.T


def search(rng, W, pairs, n, cls, c, budget, rmax=2.0):
    k = len(pairs)
    dim = k if cls == "DD" else 2 * k
    lo = np.r_[-c * np.ones(k), -rmax * np.ones(dim - k)]
    hi = np.r_[c * np.ones(k), rmax * np.ones(dim - k)]
    split = lambda x: (x[:k], x[k:] if cls == "SDD" else np.zeros(k))
    f = lambda x: -fd.spec(W, omega(loadings(*split(x), pairs, n)))[1]
    best = max((rng.uniform(lo, hi) for _ in range(budget // 4)), key=f)
    bv, step = f(best), c / 2
    for it in range(budget):
        x = best.copy(); m = rng.integers(dim)
        x[m] = np.clip(x[m] + rng.normal(0, step), lo[m], hi[m])
        v = f(x)
        if v > bv: best, bv = x, v
        if it % 300 == 299: step *= 0.7
    return loadings(*split(best), pairs, n)


def rcd_metrics(W, pairs, B):
    n = W.shape[0]
    truth = np.zeros((n, n), int)  # code for unordered pair (i<j): 0 none, 1 i->j, 2 j->i, 3 <->
    est = np.zeros((n, n), int)
    conf = np.zeros((n, n), bool)
    for i, j in pairs:
        conf[i, j] = conf[j, i] = True
    for i in range(n):
        for j in range(i + 1, n):
            truth[i, j] = 3 if conf[i, j] else 1 if W[j, i] else 2 if W[i, j] else 0
            if np.isnan(B[i, j]) or np.isnan(B[j, i]):
                est[i, j] = 3
            else:
                est[i, j] = 1 if B[j, i] != 0 else 2 if B[i, j] != 0 else 0
    iu = np.triu_indices(n, 1)
    t, e = truth[iu], est[iu]
    shd = np.mean(t != e)
    dt, de = np.isin(t, (1, 2)), np.isin(e, (1, 2))
    dir_r = np.sum(dt & (t == e)) / max(dt.sum(), 1)
    dir_p = np.sum(de & (t == e)) / max(de.sum(), 1)
    bi_r = np.sum((t == 3) & (e == 3)) / max(np.sum(t == 3), 1)
    bi_p = np.sum((t == 3) & (e == 3)) / max(np.sum(e == 3), 1)
    return [shd, dir_r, dir_p, bi_r, bi_p]


def run_arm(W, pairs, Lam, Ue, UL, A, T):
    from lingam import RCD
    from causallearn.search.ConstraintBased.FCI import fci
    Om = omega(Lam)
    X = (Ue + UL @ Lam.T) @ A.T
    X = (X - X.mean(0)) / X.std(0)
    m = RCD()
    m.fit(X)
    G, _ = fci(X, "fisherz", 0.05, verbose=False, show_progress=False)
    return list(fd.spec(W, Om)) + rcd_metrics(W, pairs, m.adjacency_matrix_) + [fd.metrics(T, G.graph)[0]]


def one(args):
    design, seed, n, wlo, whi, c, N, budget = args
    rng = np.random.default_rng(seed)
    W, pairs = fr.instance(rng, n, wlo, whi)
    k = len(pairs)
    if design == "E":
        Lams = [search(rng, W, pairs, n, cls, c, budget) for cls in ("DD", "SDD")]
    else:
        th = rng.uniform(-c, c, k)
        Lams = [loadings(th, np.zeros(k), pairs, n), loadings(th, rng.uniform(-2, 2, k), pairs, n)]
    T = fd.true_pag(W, pairs)
    A = np.linalg.inv(np.eye(n) - W)
    s3 = np.sqrt(3)
    Ue = rng.uniform(-s3, s3, (N, n))
    UL = rng.uniform(-s3, s3, (N, k))
    return np.array([run_arm(W, pairs, L, Ue, UL, A, T) for L in Lams])


if __name__ == "__main__":
    from scipy.stats import wilcoxon
    design = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    Ns = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [500, 2000]
    names = ["rho(R~)", "lmin(P)", "RCD SHD", "RCD dir R", "RCD dir P", "RCD <-> R", "RCD <-> P", "FCI SHD"]
    settings = [(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)] for c in [0.7, 2.0] for N in Ns]
    base = 40_000 if design == "E" else 50_000
    res = {}
    with Pool(16) as pool:
        for s, (wlo, whi, c, N) in enumerate(settings):
            R = np.array(pool.map(one, [(design, base * (s + 1) + r, 10, wlo, whi, c, N, 1500)
                                        for r in range(reps)]))
            key = f"W[{wlo},{whi}]_c{c}_N{N}"; res[key] = R
            print(f"{key}  design {design}, {len(R)} paired instances, means; Wilcoxon p")
            for q, nm in enumerate(names):
                dd, sd = R[:, 0, q], R[:, 1, q]
                p = wilcoxon(dd, sd).pvalue if np.any(dd != sd) else 1.0
                print(f"   {nm:10s} DD {dd.mean():.3f}  SDD {sd.mean():.3f}  p={p:.1e}", flush=True)
    np.savez(f"rcd_{design}.npz", **res)
