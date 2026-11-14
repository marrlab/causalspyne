"""Raw material for presenting the CI-test result in several ways (forest plot, recall vs strength, sample-size
equivalent, pooled ROC, paired scatter).  Both families: Gaussian (design B: DD vs explicit extreme) and non-Gaussian
(design E: DD vs root-explicit), 4 settings each, same ADMG in both arms.  Fisher-z, alpha=0.05, |S|<=2, vs exact CI.
Stores per ADMG recall/AUC at N in NS, and the per-assertion (|true pc|, z, dependent) at N=500.
Usage: python ci_present.py [instances]"""
import sys, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import faithfulness_margin as fm
import rcd_dd_vs_sdd as rd, fci_rejection as fr
from ci_test_performance import keys_for, pcs, Z
from ci_auc import auc

NS = [250, 350, 500, 700, 1000, 1400, 2000, 2800, 4000, 5000]
I500 = NS.index(500)
SETS = [(0.1, 0.5, 0.7), (0.1, 0.5, 2.0), (0.5, 1.5, 0.7), (0.5, 1.5, 2.0)]


def evaluate(Xs, ks, sizes, dep, agree, pop):
    """Xs: list over arms of (5000 x n) data. returns metrics (arm, N, [recall, auc]) and the N=500 raw arrays"""
    res = np.zeros((2, len(NS), 2)); raw = []
    for a_, X in enumerate(Xs):
        for j, N in enumerate(NS):
            ph = pcs(np.corrcoef(X[:N].T), ks)
            z = np.abs(np.arctanh(np.clip(ph, -0.999999, 0.999999))) * np.sqrt(N - sizes - 3)
            res[a_, j] = [(z > Z)[dep & agree].mean(), auc(z[agree], dep[agree])]
            if N == 500:
                raw.append(np.stack([z[agree], np.abs(pop[a_])[agree], dep[agree].astype(float)]))
    return res, raw


def one_g(a):
    s, r = a
    wlo, whi, c = SETS[s]
    W, pairs, Oms = fm.inst_B(20_000 * (2 * s + 1) + r, 10, wlo, whi, c, 5000)
    n = W.shape[0]; ks = keys_for(n); A = np.linalg.inv(np.eye(n) - W)
    sizes = np.array([len(k[2]) for k in ks])
    pops = [pcs(A @ Om @ A.T, ks) for Om in Oms]
    d0, d1 = np.abs(pops[0]) > 1e-9, np.abs(pops[1]) > 1e-9
    Zs = np.random.default_rng(777_000 + 100 * s + r).standard_normal((5000, n))
    Xs = [Zs @ np.linalg.cholesky(Om).T @ A.T for Om in Oms]
    return evaluate(Xs, ks, sizes, d0, d0 == d1, pops)


def one_n(a):
    s, r = a
    wlo, whi, c = SETS[s]; n = 10
    rng = np.random.default_rng(40_000 * (s + 1) + r)
    W, pairs = fr.instance(rng, n, wlo, whi); k = len(pairs)
    Lams = [rd.search(rng, W, pairs, n, cls, c, 1500) for cls in ("DD", "SDD")]
    A = np.linalg.inv(np.eye(n) - W); ks = keys_for(n); sizes = np.array([len(q[2]) for q in ks])
    s3 = np.sqrt(3)
    Ue = rng.uniform(-s3, s3, (5000, n)); UL = rng.uniform(-s3, s3, (5000, k))
    pops = [pcs(A @ rd.omega(L) @ A.T, ks) for L in Lams]
    d0, d1 = np.abs(pops[0]) > 1e-9, np.abs(pops[1]) > 1e-9
    Xs = [(Ue + UL @ L.T) @ A.T for L in Lams]
    return evaluate(Xs, ks, sizes, d0, d0 == d1, pops)


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    store = {}
    with Pool(16) as pool:
        for fam, fn in (("G", one_g), ("N", one_n)):
            for s in range(len(SETS)):
                out = pool.map(fn, [(s, r) for r in range(R)])
                store[f"{fam}{s}_metrics"] = np.array([o[0] for o in out])            # R x arm x N x 2
                store[f"{fam}{s}_raw"] = np.concatenate([np.concatenate(o[1], axis=1) for o in out], axis=1) if False else None
                # pooled per-assertion arrays per arm: (3, M)
                for a_ in (0, 1):
                    store[f"{fam}{s}_raw{a_}"] = np.concatenate([o[1][a_] for o in out], axis=1)
                print(fam, s, "done", flush=True)
    store = {k: v for k, v in store.items() if v is not None}
    np.savez("ci_present.npz", NS=np.array(NS), SETS=np.array(SETS), **store)
