"""Does diagonal dominance of Omega change FCI's measured performance, for a FIXED graph?

For each instance: one DAG W over n observed variables and one sparse confounding pattern
(bidirected pairs, each variable in <= 2 pairs). Off-diagonal Omega entries theta_ij ~ U(+-[c/4, c]).
Two arms share W and theta exactly; only the diagonal of Omega differs:
  DD : Omega_ii = 1 + sum_j |theta_ij|            (the paper's construction)
  PD : Omega = Theta + (1 - lambda_min(Theta)) I  (smallest diagonal with lambda_min(Omega) = 1)
So both arms have the same ADMG, hence the same true PAG, and the same noise floor.
Data X = (I - W)^{-1} eps, eps ~ N(0, Omega). FCI (fisherz) vs true PAG (dag2pag with one latent per
bidirected pair). Reports normalized SHD, adjacency P/R and arrowhead P/R, plus rho(R~) and
lambda_min(P), paired over instances.
"""
import argparse
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore")


def true_pag(W, pairs):
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.DAG2PAG import dag2pag
    n, k = W.shape[0], len(pairs)
    nodes = [GraphNode(f"X{i}") for i in range(n + k)]
    g = Dag(nodes)
    for ch, pa in np.argwhere(W != 0):
        g.add_directed_edge(nodes[pa], nodes[ch])
    for l, (i, j) in enumerate(pairs):
        g.add_directed_edge(nodes[n + l], nodes[i])
        g.add_directed_edge(nodes[n + l], nodes[j])
    pag = dag2pag(g, nodes[n:])
    # dag2pag scrambles the node order: map back to X0..X{n-1}
    idx = [int(v.get_name()[1:]) for v in pag.get_nodes()]
    T = np.zeros((n, n), int)
    T[np.ix_(idx, idx)] = pag.graph
    return T


def spec(W, Om):
    n = W.shape[0]
    A = np.eye(n) - W
    P = A.T @ np.linalg.solve(Om, A)
    d = 1 / np.sqrt(np.diag(P))
    ev = np.linalg.eigvalsh(d[:, None] * P * d[None, :])
    return ev[-1], ev[0]


def metrics(T, G):
    n = T.shape[0]
    off = ~np.eye(n, dtype=bool)
    shd = np.sum(T != G) / (n * n - n)
    at, ag = (T != 0) & off, (G != 0) & off
    tp = np.sum(at & ag) / 2
    adj_p = tp / max(ag.sum() / 2, 1)
    adj_r = tp / max(at.sum() / 2, 1)
    ht, hg = (T == 1), (G == 1)  # arrowhead at row-node
    ah_p = np.sum(ht & hg) / max(hg.sum(), 1)
    ah_r = np.sum(ht & hg) / max(ht.sum(), 1)
    return shd, adj_p, adj_r, ah_p, ah_r


def one(args):
    seed, n, wlo, whi, c, N, alpha = args
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
    Th = np.zeros((n, n))
    for i, j in pairs:
        Th[i, j] = Th[j, i] = rng.uniform(c / 4, c) * rng.choice([-1, 1])
    Om = {"DD": Th + np.diag(1 + np.abs(Th).sum(1)),
          "PD": Th + (1 - np.linalg.eigvalsh(Th)[0]) * np.eye(n)}
    T = true_pag(W, pairs)
    A = np.linalg.inv(np.eye(n) - W)
    Z = rng.standard_normal((N, n))  # common random numbers across arms
    out = []
    for arm in ("DD", "PD"):
        L = np.linalg.cholesky(Om[arm])
        X = Z @ L.T @ A.T
        X = (X - X.mean(0)) / X.std(0)
        G, _ = fci(X, "fisherz", alpha, verbose=False, show_progress=False)
        out.append(spec(W, Om[arm]) + metrics(T, G.graph))
    return np.array(out)  # 2 x 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", default="fci_dd_vs_pd.npz")
    a = ap.parse_args()
    from scipy.stats import wilcoxon
    names = ["rho(R~)", "lmin(P)", "SHD", "adj P", "adj R", "arrow P", "arrow R"]
    res = {}
    settings = [(wlo, whi, c, N) for (wlo, whi) in [(0.1, 0.5), (0.5, 1.5)]
                for c in [0.7, 2.0] for N in [500, 5000]]
    with Pool(a.procs) as pool:
        for s, (wlo, whi, c, N) in enumerate(settings):
            jobs = [(10_000 * s + r, a.n, wlo, whi, c, N, a.alpha) for r in range(a.reps)]
            R = np.array(pool.map(one, jobs))  # reps x 2 x 7
            key = f"W[{wlo},{whi}]_c{c}_N{N}"
            res[key] = R
            print(f"{key}  (n={a.n}, {a.reps} paired instances, means; p = Wilcoxon signed-rank)")
            for k, nm in enumerate(names):
                dd, pd = R[:, 0, k], R[:, 1, k]
                p = wilcoxon(dd, pd).pvalue if np.any(dd != pd) else 1.0
                print(f"   {nm:8s} DD {dd.mean():.3f}  PD {pd.mean():.3f}  p={p:.1e}")
            print(flush=True)
    np.savez(a.out, **res)


if __name__ == "__main__":
    main()
