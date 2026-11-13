"""Faithfulness margin of the F.6 instances, and whether it explains FCI's SHD.

FCI with Fisher-z depends on the data only through the sample correlation matrix, and its tests use
partial correlations given small conditioning sets. The zero pattern is fixed by the ADMG (same in both
arms); what differs is how weak the non-zero dependencies are. For each instance we compute, exactly
from Sigma = (I-W)^{-1} Omega (I-W)^{-T}:
  per adjacent pair (i,j) of the true MAG: m_ij = min_{S, |S|<=3} |pcorr(i,j | S)|
  margin_min  = min over adjacent pairs of m_ij        (weakest dependence FCI must detect)
  margin_mean = mean over adjacent pairs of m_ij
  n_weak      = number of adjacent pairs with m_ij below the Fisher-z detection level at N
                (z = atanh(m) * sqrt(N - |S| - 3) < 1.96, using |S| = 3)
The Omega of each instance is regenerated from the same seeds and random streams as fci_dd_vs_pd.py,
fci_extremes.py and fci_rejection.py (checked against the stored rho(R~)); SHD is read from their npz.
Usage: python faithfulness_margin.py
"""
import itertools, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import fixedW_extremes as fx
import fci_dd_vs_pd as fd
import fci_extremes as fe
import fci_rejection as fr

SETTINGS = [(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)] for c in [0.7, 2.0] for N in [500, 5000]]


def adjacent_pairs(W, pairs):
    # adjacencies of the true PAG (= true MAG skeleton)
    T = fd.true_pag(W, pairs)
    n = W.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if T[i, j] != 0]


def margins(W, Om, adj, N, smax=3):
    n = W.shape[0]
    A = np.linalg.inv(np.eye(n) - W)
    S = A @ Om @ A.T
    m = []
    for i, j in adj:
        rest = [k for k in range(n) if k not in (i, j)]
        best = np.inf
        for s in range(smax + 1):
            for cs in itertools.combinations(rest, s):
                idx = [i, j, *cs]
                Pm = np.linalg.inv(S[np.ix_(idx, idx)])
                pc = -Pm[0, 1] / np.sqrt(Pm[0, 0] * Pm[1, 1])
                best = min(best, abs(pc))
        m.append(best)
    m = np.array(m)
    z = np.arctanh(np.minimum(m, 0.999999)) * np.sqrt(N - smax - 3)
    return [m.min(), m.mean(), np.sum(z < 1.96)]


def inst_A(seed, n, wlo, whi, c):
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n)); msk = np.tril(rng.random((n, n)) < 0.3, -1)
    W[msk] = rng.uniform(wlo, whi, msk.sum()) * rng.choice([-1, 1], msk.sum())
    perm = rng.permutation(n); W = W[np.ix_(perm, perm)]
    pairs, deg = [], np.zeros(n, int)
    for _ in range(4 * n):
        i, j = sorted(rng.choice(n, 2, replace=False))
        if deg[i] < 2 and deg[j] < 2 and (i, j) not in pairs and W[i, j] == 0 and W[j, i] == 0:
            pairs.append((i, j)); deg[i] += 1; deg[j] += 1
    Th = np.zeros((n, n))
    for i, j in pairs:
        Th[i, j] = Th[j, i] = rng.uniform(c / 4, c) * rng.choice([-1, 1])
    return W, pairs, [Th + np.diag(1 + np.abs(Th).sum(1)), Th + (1 - np.linalg.eigvalsh(Th)[0]) * np.eye(n)]


def inst_B(seed, n, wlo, whi, c, N):
    rng = np.random.default_rng(seed)
    W, pairs = fr.instance(rng, n, wlo, whi)
    rng.standard_normal((N, n))  # Z in fci_extremes.one
    return W, pairs, [fe.argsearch(rng, W, pairs, n, cls, c, 1500) for cls in ("DD", "PD")]


def inst_C(seed, n, wlo, whi, c):
    rng = np.random.default_rng(seed)
    W, pairs = fr.instance(rng, n, wlo, whi)
    k = len(pairs)
    fx.search(rng, W, pairs, n, "DD", "rho", c, 1500)
    dd_rand = [fd.spec(W, fx.build(rng.uniform(-c, c, k), None, pairs, n, "DD"))[0] for _ in range(2000)]
    thr = np.quantile(dd_rand, 0.95)
    pd_om = None
    for t in range(20000):
        th = rng.uniform(-c, c, k) * (rng.random(k) < 1)
        dv = rng.exponential(0.3, n)
        Om = fx.build(th, dv, pairs, n, "PD")
        if fd.spec(W, Om)[0] > thr:
            pd_om = Om; break
    if pd_om is None:
        return None
    return W, pairs, [fx.build(rng.uniform(-c, c, k), None, pairs, n, "DD"), pd_om]


def one(args):
    design, s, r = args
    wlo, whi, c, N = SETTINGS[s]
    if design == "A":
        inst = inst_A(10_000 * s + r, 10, wlo, whi, c)
    elif design == "B":
        inst = inst_B(20_000 * s + r, 10, wlo, whi, c, N)
    else:
        inst = inst_C(30_000 * s + r, 10, wlo, whi, c)
    if inst is None:
        return None
    W, pairs, Oms = inst
    adj = adjacent_pairs(W, pairs)
    return np.array([[fd.spec(W, Om)[0]] + margins(W, Om, adj, N) for Om in Oms])


if __name__ == "__main__":
    from scipy.stats import wilcoxon, spearmanr
    files = {"A": ("fci_dd_vs_pd.npz", 200), "B": ("fci_extremes.npz", 150), "C": ("fci_rejection.npz", 150)}
    with Pool(16) as pool:
        for design, (f, reps) in files.items():
            D = np.load(f, allow_pickle=True)
            print(f"=== design {design}")
            for s, (wlo, whi, c, N) in enumerate(SETTINGS):
                key = f"W[{wlo},{whi}]_c{c}_N{N}"
                M = [m for m in pool.map(one, [(design, s, r) for r in range(reps)]) if m is not None]
                M = np.array(M)  # inst x 2 x 4 (rho, margin_min, margin_mean, n_weak)
                R = D[key]
                chk = np.max(np.abs(M[:, :, 0] - R[:, :, 0]))
                shd = R[:, :, 2]
                dm = M[:, 1, 2] - M[:, 0, 2]; dshd = shd[:, 1] - shd[:, 0]
                pool_r = spearmanr(M[:, :, 2].ravel(), shd.ravel())[0]
                wk_r = spearmanr(M[:, :, 3].ravel(), shd.ravel())[0]
                within = spearmanr(dm, dshd)
                p_mean = wilcoxon(M[:, 0, 2], M[:, 1, 2]).pvalue
                print(f"{key}  (rho check max err {chk:.1e})\n"
                      f"   margin min  DD {np.median(M[:,0,1]):.3f}  PD {np.median(M[:,1,1]):.3f} (medians)\n"
                      f"   margin mean DD {M[:,0,2].mean():.3f}  PD {M[:,1,2].mean():.3f}  p={p_mean:.1e}\n"
                      f"   weak pairs  DD {M[:,0,3].mean():.2f}  PD {M[:,1,3].mean():.2f}\n"
                      f"   Spearman pooled: margin_mean~SHD {pool_r:+.2f}, n_weak~SHD {wk_r:+.2f};"
                      f" within pair d(margin_mean)~d(SHD) {within[0]:+.2f} (p={within[1]:.2f})", flush=True)
