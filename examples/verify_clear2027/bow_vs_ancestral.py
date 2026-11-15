"""Ancestral (bow-free) vs bow-containing confounding, same fixed observed graph, same ancestral ground truth.
Arm A: hidden roots only on non-adjacent pairs (the ancestral ADMG's bidirected edges).
Arm B: arm A plus one extra hidden root on each of k random directed edges i->j (a bow: i->j together with i<->j);
       k = number of bidirected pairs (capped by the number of directed edges). Same W, same theta on the shared pairs,
       same noise draws. Loadings: DD construction (a = b = sqrt|theta|), uniform unit-variance noise (non-Gaussian).
Score: FCI (fisherz) normalized SHD and RCD pair-SHD against the ancestral truth T_A (the MAG/PAG of arm A; the
ground truth used for evaluation), plus how often the PAG of the DAG with the bow latents equals T_A.
Usage: python bow_vs_ancestral.py reps"""
import sys, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import rcd_dd_vs_sdd as r
import fci_dd_vs_pd as fd
import fci_rejection as fr

def one(args):
    seed, n, wlo, whi, c, N = args
    from lingam import RCD
    from causallearn.search.ConstraintBased.FCI import fci
    rng = np.random.default_rng(seed)
    W, pairs = fr.instance(rng, n, wlo, whi)
    edges = [(pa, ch) for ch, pa in np.argwhere(W != 0)]  # pa -> ch
    kb = min(len(pairs), len(edges))
    if not pairs or kb == 0:
        return None
    sel = rng.choice(len(edges), kb, replace=False)
    bows = [tuple(sorted(edges[s])) for s in sel]
    th = rng.uniform(-c, c, len(pairs)); thb = rng.uniform(-c, c, kb)
    LamA = r.loadings(th, np.zeros(len(pairs)), pairs, n)
    LamB = np.hstack([LamA, r.loadings(thb, np.zeros(kb), bows, n)])
    TA = fd.true_pag(W, pairs)
    TB = fd.true_pag(W, pairs + bows)
    same = float(np.array_equal(TA, TB))
    A = np.linalg.inv(np.eye(n) - W); s3 = np.sqrt(3)
    Ue = rng.uniform(-s3, s3, (N, n)); UL = rng.uniform(-s3, s3, (N, len(pairs) + kb))
    out = []
    for Lam, Town in ((LamA, TA), (LamB, TB)):
        X = (Ue + UL[:, :Lam.shape[1]] @ Lam.T) @ A.T
        X = (X - X.mean(0)) / X.std(0)
        m = RCD(); m.fit(X)
        G, _ = fci(X, "fisherz", 0.05, verbose=False, show_progress=False)
        out.append([fd.metrics(TA, G.graph)[0], r.rcd_metrics(W, pairs, m.adjacency_matrix_)[0], fd.spec(W, r.omega(Lam))[0], fd.metrics(Town, G.graph)[0]])
    return np.array(out), same

if __name__ == "__main__":
    from scipy.stats import wilcoxon
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    names = ["FCI SHD vs T_A", "RCD pair-SHD", "rho(R~)", "FCI SHD vs own"]
    res = {}
    settings = [(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)] for c in [0.7, 2.0] for N in [500, 2000]]
    with Pool(16) as pool:
        for s, (wlo, whi, c, N) in enumerate(settings):
            rows = [x for x in pool.map(one, [(70_000 * (s + 1) + k, 10, wlo, whi, c, N) for k in range(reps)]) if x is not None]
            R = np.array([x[0] for x in rows]); same = np.mean([x[1] for x in rows])
            key = f"W[{wlo},{whi}]_c{c}_N{N}"; res[key] = R
            print(f"{key}  {len(R)} paired ADMGs, PAG(with bows)==PAG(ancestral) in {same:.0%}; means, Wilcoxon p")
            for q, nm in enumerate(names):
                a, b = R[:, 0, q], R[:, 1, q]
                p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
                print(f"   {nm:16s} ancestral {a.mean():.3f}  with bows {b.mean():.3f}  p={p:.1e}", flush=True)
    np.savez("bow_vs_ancestral.npz", **res)
