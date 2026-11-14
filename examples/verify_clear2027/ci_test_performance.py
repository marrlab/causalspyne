"""Sample-level performance of the CI tests FCI relies on, DD vs the explicit extreme (PD), same ADMG.

Design B instances (same W and bidirected pattern; Omega = DD or the explicit extreme with the smallest lambda_min(P)),
the same standard normal draws in both arms.  For every assertion i _||_ j | S with |S| <= 2 the true statement is known
exactly (partial correlation zero or not, from Sigma).  The Fisher-z test (alpha = 0.05) is run on the sample
correlation matrix of N standardized observations.  Per instance and arm:
  recall      = rejected / dependent assertions            (power; the true dependencies that are detected)
  specificity = not rejected / independent assertions
  accuracy    = correct decisions / all assertions
  weak recall = recall on the dependent assertions with |pc| < 0.1 under BOTH arms' population value (hard cases)
Output: ci_test_performance.npz (arrays, instances x arm x metric) and ci_test_performance_output.txt.
Usage: python ci_test_performance.py [instances]
"""
import itertools, sys, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import faithfulness_margin as fm

SET = [(0.1, 0.5, 0.7), (0.1, 0.5, 2.0), (0.5, 1.5, 0.7), (0.5, 1.5, 2.0)]
NS = (500, 5000)
Z = 1.959964


def keys_for(n, smax=2):
    ks = []
    for i in range(n):
        for j in range(i + 1, n):
            rest = [k for k in range(n) if k not in (i, j)]
            for s in range(smax + 1):
                for cs in itertools.combinations(rest, s):
                    ks.append((i, j, cs))
    return ks


def pcs(S, ks):
    out = np.empty(len(ks))
    for q, (i, j, cs) in enumerate(ks):
        idx = [i, j, *cs]
        P = np.linalg.inv(S[np.ix_(idx, idx)])
        out[q] = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
    return out


def one(a):
    s, r = a
    wlo, whi, c = SET[s]
    W, pairs, Oms = fm.inst_B(20_000 * (2 * s + 1) + r, 10, wlo, whi, c, 5000)
    n = W.shape[0]
    ks = keys_for(n)
    A = np.linalg.inv(np.eye(n) - W)
    sizes = np.array([len(k[2]) for k in ks])
    pops = [pcs(A @ Om @ A.T, ks) for Om in Oms]
    dep = [np.abs(p) > 1e-9 for p in pops]
    weak = dep[0] & dep[1] & (np.abs(pops[0]) < 0.1) & (np.abs(pops[1]) < 0.1)
    rng = np.random.default_rng(777_000 + 100 * s + r)
    Zs = rng.standard_normal((max(NS), n))
    res = np.zeros((len(NS), 2, 4))
    for a_, Om in enumerate(Oms):
        L = np.linalg.cholesky(Om)
        Xall = Zs @ L.T @ A.T
        for b_, N in enumerate(NS):
            X = Xall[:N]
            Sh = np.corrcoef(X.T)
            ph = pcs(Sh, ks)
            rej = np.abs(np.arctanh(np.clip(ph, -0.999999, 0.999999))) * np.sqrt(N - sizes - 3) > Z
            d = dep[a_]
            res[b_, a_] = [rej[d].mean(), (~rej[~d]).mean() if (~d).any() else np.nan,
                           np.mean(rej == d), rej[d & weak].mean() if (d & weak).any() else np.nan]
    return res


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    from scipy.stats import wilcoxon
    names = ["recall", "specificity", "accuracy", "weak recall"]
    allres = {}
    lines = []
    with Pool(16) as pool:
        for s, (wlo, whi, c) in enumerate(SET):
            M = np.array(pool.map(one, [(s, r) for r in range(R)]))  # R x N x arm x metric
            allres[f"s{s}"] = M
            for b_, N in enumerate(NS):
                lines.append(f"W[{wlo},{whi}] c={c} N={N} ({R} instances)")
                for m, nm in enumerate(names):
                    x, y = M[:, b_, 0, m], M[:, b_, 1, m]
                    ok = ~(np.isnan(x) | np.isnan(y))
                    p = wilcoxon(x[ok], y[ok]).pvalue if ok.sum() > 5 and np.any(x[ok] != y[ok]) else float("nan")
                    lines.append(f"   {nm:12s} DD {np.nanmean(x):.3f}  PD {np.nanmean(y):.3f}  diff {np.nanmean(y - x):+.3f}  p={p:.1e}")
                print("\n".join(lines[-5:]), flush=True)
    np.savez("ci_test_performance.npz", sets=np.array(SET), ns=np.array(NS), **allres)
    open("ci_test_performance_output.txt", "w").write("\n".join(lines) + "\n")
