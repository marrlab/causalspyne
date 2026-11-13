"""Downstream effects of the DD restriction beyond FCI, same fixed W and bidirected pattern.

Omega pairs are drawn exactly as in fci_extremes.py (design B: per class the Omega minimizing
lambda_min(P)) and fci_rejection.py (design C: random DD vs PD rejection-sampled above the DD 95th
percentile of rho(R~)), with the same seeds, so the instances match F.6.
Measures, per instance and arm:
  * effect estimation: population OLS of X_u on its true parents (the correct adjustment set if there
    were no hidden confounding). Bias of each edge coefficient = [Sigma_pa^-1 Cov(X_pa, eps_u)].
    Reported: mean |bias| over edges, max |bias|, fraction of edges with |bias| > 0.1|W_uv|... (rel > 10%),
    and the fraction of edges whose estimated sign is wrong.
  * GES (BIC, Gaussian, confounder-unaware) on standardized data: spurious adjacency rate among
    non-adjacent pairs of W, spurious-adjacency rate among bidirected pairs, adjacency recall.
Usage: python downstream_dd_vs_pd.py B|C reps
"""
import sys, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import fixedW_extremes as fx
import fci_dd_vs_pd as fd
import fci_extremes as fe
import fci_rejection as fr


def effect_bias(W, Om):
    n = W.shape[0]
    A = np.linalg.inv(np.eye(n) - W)
    S = A @ Om @ A.T
    C = A @ Om  # Cov(X, eps)
    ab, rel, sign = [], [], []
    for u in range(n):
        pa = np.flatnonzero(W[u])
        if len(pa) == 0:
            continue
        b = np.linalg.solve(S[np.ix_(pa, pa)], C[pa, u])
        ab += list(np.abs(b)); rel += list(np.abs(b) / np.abs(W[u, pa]))
        sign += list(np.sign(W[u, pa] + b) != np.sign(W[u, pa]))
    if not ab:
        return [np.nan] * 4
    ab, rel = np.array(ab), np.array(rel)
    return [ab.mean(), ab.max(), np.mean(rel > 0.1), np.mean(sign)]


def ges_metrics(W, pairs, X):
    from causallearn.search.ScoreBased.GES import ges
    n = W.shape[0]
    G = ges(X, score_func="local_score_BIC")["G"].graph
    adj_g = (G != 0) | (G.T != 0)
    adj_t = (W != 0) | (W.T != 0)
    iu = np.triu_indices(n, 1)
    ag, at = adj_g[iu], adj_t[iu]
    bi = np.zeros((n, n), bool)
    for i, j in pairs:
        bi[i, j] = bi[j, i] = True
    b = bi[iu]
    fp = np.sum(ag & ~at) / max(np.sum(~at), 1)
    fp_bi = np.sum(ag & b) / max(b.sum(), 1)
    rec = np.sum(ag & at) / max(at.sum(), 1)
    return [fp, fp_bi, rec]


def run_arm(W, pairs, Om, Z, A):
    X = Z @ np.linalg.cholesky(Om).T @ A.T
    X = (X - X.mean(0)) / X.std(0)
    return list(fd.spec(W, Om)) + effect_bias(W, Om) + ges_metrics(W, pairs, X)


def one_B(args):
    seed, n, wlo, whi, c, N, budget = args
    rng = np.random.default_rng(seed)
    W, pairs = fr.instance(rng, n, wlo, whi)  # identical draws to fci_extremes.one
    fd.true_pag  # (PAG not needed)
    A = np.linalg.inv(np.eye(n) - W)
    Z = rng.standard_normal((N, n))
    out = []
    for cls in ("DD", "PD"):
        Om = fe.argsearch(rng, W, pairs, n, cls, c, budget)
        out.append(run_arm(W, pairs, Om, Z, A))
    return np.array(out)


def one_C(args):
    seed, n, wlo, whi, c, N, tries = args
    rng = np.random.default_rng(seed)
    W, pairs = fr.instance(rng, n, wlo, whi)
    k = len(pairs)
    fx.search(rng, W, pairs, n, "DD", "rho", c, 1500)  # keep the rng stream identical to F.6
    dd_rand = [fd.spec(W, fx.build(rng.uniform(-c, c, k), None, pairs, n, "DD"))[0] for _ in range(2000)]
    thr = np.quantile(dd_rand, 0.95)
    pd_om = None
    for t in range(tries):
        th = rng.uniform(-c, c, k) * (rng.random(k) < 1)
        dv = rng.exponential(0.3, n)
        Om = fx.build(th, dv, pairs, n, "PD")
        if fd.spec(W, Om)[0] > thr:
            pd_om = Om; break
    if pd_om is None:
        return None
    dd_om = fx.build(rng.uniform(-c, c, k), None, pairs, n, "DD")
    A = np.linalg.inv(np.eye(n) - W)
    Z = rng.standard_normal((N, n))
    return np.array([run_arm(W, pairs, Om, Z, A) for Om in (dd_om, pd_om)])


if __name__ == "__main__":
    from scipy.stats import wilcoxon
    design = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    names = ["rho(R~)", "lmin(P)", "eff |bias| mean", "eff |bias| max", "eff rel>10%",
             "eff sign wrong", "GES FP adj", "GES FP on <->", "GES adj R"]
    settings = [(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)] for c in [0.7, 2.0] for N in [500, 5000]]
    res = {}
    with Pool(16) as pool:
        for s, (wlo, whi, c, N) in enumerate(settings):
            if design == "B":
                R = pool.map(one_B, [(20_000 * s + r, 10, wlo, whi, c, N, 1500) for r in range(reps)])
            else:
                R = pool.map(one_C, [(30_000 * s + r, 10, wlo, whi, c, N, 20000) for r in range(reps)])
            R = np.array([x for x in R if x is not None])
            key = f"W[{wlo},{whi}]_c{c}_N{N}"; res[key] = R
            print(f"{key}  design {design}, {len(R)} paired instances, means; Wilcoxon p")
            for k, nm in enumerate(names):
                dd, pd = R[:, 0, k], R[:, 1, k]
                ok = ~(np.isnan(dd) | np.isnan(pd))
                dd, pd = dd[ok], pd[ok]
                p = wilcoxon(dd, pd).pvalue if np.any(dd != pd) else 1.0
                print(f"   {nm:16s} DD {dd.mean():.3f}  PD {pd.mean():.3f}  p={p:.1e}", flush=True)
    np.savez(f"downstream_{design}.npz", **res)
