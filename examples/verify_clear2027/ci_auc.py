"""CI assertions as a random variable: is there a true DD vs PD difference?  (one sample size, N=500)

Same instances as ci_test_performance.py (design B: same W and bidirected pattern, Omega = DD vs explicit extreme).
Unit of analysis = one ADMG (assertions within an ADMG are dependent).  Per ADMG and arm:
  recall, accuracy      at alpha=0.05
  AUC                   of the Fisher-z statistic |z| as a score for 'assertion is truly dependent' (threshold free)
  AUC_weak              the same restricted to assertions with |pc|<0.1 in both arms (hard cases)
Paired test over ADMGs: Wilcoxon signed-rank; report mean difference PD-DD, 95% bootstrap CI, p.
Usage: python ci_auc.py [instances]
"""
import sys, numpy as np, warnings
from multiprocessing import Pool
from scipy.stats import rankdata, wilcoxon
warnings.filterwarnings("ignore")
import faithfulness_margin as fm
from ci_test_performance import SET, keys_for, pcs, Z

N = 500


def auc(score, lab):
    n1, n0 = lab.sum(), (~lab).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def one(a):
    s, r = a
    wlo, whi, c = SET[s]
    W, pairs, Oms = fm.inst_B(20_000 * (2 * s + 1) + r, 10, wlo, whi, c, 5000)
    n = W.shape[0]
    ks = keys_for(n)
    A = np.linalg.inv(np.eye(n) - W)
    sizes = np.array([len(k[2]) for k in ks])
    pops = [pcs(A @ Om @ A.T, ks) for Om in Oms]
    d0, d1 = np.abs(pops[0]) > 1e-9, np.abs(pops[1]) > 1e-9
    # true CI statements are identical for both arms (same ADMG); assertions where float noise makes the arms disagree are dropped
    agree = d0 == d1
    dep = d0
    weak = dep & (np.abs(pops[0]) < 0.1) & (np.abs(pops[1]) < 0.1)
    rng = np.random.default_rng(777_000 + 100 * s + r)
    Zs = rng.standard_normal((5000, n))
    out = np.zeros((2, 5))
    for a_, Om in enumerate(Oms):
        X = (Zs @ np.linalg.cholesky(Om).T @ A.T)[:N]
        ph = pcs(np.corrcoef(X.T), ks)
        z = np.abs(np.arctanh(np.clip(ph, -0.999999, 0.999999))) * np.sqrt(N - sizes - 3)
        rej = z > Z
        keep = (~dep) | weak                 # AUC_weak: independents vs weak dependents
        out[a_] = [rej[dep & agree].mean(), np.mean((rej == dep)[agree]), auc(z[agree], dep[agree]), auc(z[keep & agree], dep[keep & agree]),
                   np.nan if not weak.any() else rej[weak].mean()]
    return out


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    names = ["recall", "accuracy", "AUC", "AUC (weak dependencies only)", "recall (weak dependencies)"]
    lines, store = [], {}
    rng = np.random.default_rng(0)
    with Pool(16) as pool:
        for s, (wlo, whi, c) in enumerate(SET):
            M = np.array(pool.map(one, [(s, r) for r in range(R)]))  # R x arm x metric
            store[f"s{s}"] = M
            lines.append(f"W[{wlo},{whi}] c={c} N={N} ({R} ADMGs, paired)")
            for m, nm in enumerate(names):
                x, y = M[:, 0, m], M[:, 1, m]
                ok = ~(np.isnan(x) | np.isnan(y)); d = (y - x)[ok]
                bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
                p = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
                lines.append(f"   {nm:30s} DD {x[ok].mean():.3f}  PD {y[ok].mean():.3f}  diff {d.mean():+.4f} "
                             f"[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]  p={p:.1e}")
            print("\n".join(lines[-6:]), flush=True)
    np.savez("ci_auc.npz", sets=np.array(SET), **store)
    open("ci_auc_output.txt", "w").write("\n".join(lines) + "\n")
