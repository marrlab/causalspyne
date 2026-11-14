"""CI-test recall and AUC on NON-Gaussian data, DD vs root-explicit (SDD), same ADMG (design E of rcd_dd_vs_sdd.py).
Uniform noise and latents (unit variance); Fisher-z test, N=500, assertions |S|<=2, scored against the exact CI statements.
Usage: python ci_auc_nongauss.py [instances]"""
import sys, numpy as np, warnings
from multiprocessing import Pool
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
import rcd_dd_vs_sdd as rd, fci_rejection as fr
from ci_test_performance import keys_for, pcs, Z
from ci_auc import auc
N = 500
SET = [(0.1, 0.5, 0.7), (0.1, 0.5, 2.0), (0.5, 1.5, 0.7), (0.5, 1.5, 2.0)]

def one(a):
    s, r = a
    wlo, whi, c = SET[s]
    n = 10
    rng = np.random.default_rng(40_000 * (s + 1) + r)
    W, pairs = fr.instance(rng, n, wlo, whi)
    k = len(pairs)
    Lams = [rd.search(rng, W, pairs, n, cls, c, 1500) for cls in ("DD", "SDD")]
    A = np.linalg.inv(np.eye(n) - W)
    ks = keys_for(n); sizes = np.array([len(q[2]) for q in ks])
    s3 = np.sqrt(3)
    Ue = rng.uniform(-s3, s3, (N, n)); UL = rng.uniform(-s3, s3, (N, k))
    pops = [pcs(A @ rd.omega(L) @ A.T, ks) for L in Lams]
    d0, d1 = np.abs(pops[0]) > 1e-9, np.abs(pops[1]) > 1e-9
    agree = d0 == d1; dep = d0
    out = np.zeros((2, 3))
    for a_, L in enumerate(Lams):
        X = (Ue + UL @ L.T) @ A.T
        ph = pcs(np.corrcoef(X.T), ks)
        z = np.abs(np.arctanh(np.clip(ph, -0.999999, 0.999999))) * np.sqrt(N - sizes - 3)
        rej = z > Z
        out[a_] = [rej[dep & agree].mean(), np.mean((rej == dep)[agree]), auc(z[agree], dep[agree])]
    return out

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    names = ["recall", "accuracy", "AUC"]; lines = []; store = {}
    rng = np.random.default_rng(0)
    with Pool(16) as pool:
        for s, (wlo, whi, c) in enumerate(SET):
            M = np.array(pool.map(one, [(s, r) for r in range(R)])); store[f"s{s}"] = M
            lines.append(f"non-Gaussian W[{wlo},{whi}] c={c} N={N} ({R} ADMGs, paired, SDD vs DD)")
            for m, nm in enumerate(names):
                x, y = M[:, 0, m], M[:, 1, m]; d = y - x
                bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
                p = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
                lines.append(f"   {nm:9s} DD {x.mean():.3f}  SDD {y.mean():.3f}  diff {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]  p={p:.1e}")
            print("\n".join(lines[-4:]), flush=True)
    np.savez("ci_auc_nongauss.npz", **store)
    open("ci_auc_nongauss_output.txt", "w").write("\n".join(lines) + "\n")
