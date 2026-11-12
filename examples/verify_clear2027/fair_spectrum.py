"""Fair comparison for Appendix F.1 of the CLeaR 2027 paper.

For each random instance we fix ONE DAG W on the observables and ONE confounding
pattern, and change only Omega:
  explicit : Omega_exp = Lambda Lambda^T + I     (root confounders, Prop. 3)
  diag-dom : Omega_dd  = same off-diagonals as Omega_exp,
             diagonal  = 1 + sum_j |off-diagonal|  (the paper's construction)
So the two arms share W, graph size, edge weights and which pairs are confounded.
We report rho(R~) = largest eigenvalue of the normalized precision matrix, its
smallest eigenvalue, the mean squared partial correlation, and the Theorem 2
bound kappa(Omega) * min(1 + d_in_max, ||I - W||_2^2).
"""
import argparse
import numpy as np


def random_dag(rng, n, p_edge, w_low, w_high):
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            if rng.random() < p_edge:
                W[i, j] = rng.uniform(w_low, w_high) * rng.choice([-1, 1])
    perm = rng.permutation(n)
    return W[np.ix_(perm, perm)]


def random_loading(rng, n, n_conf, children, l_low, l_high):
    Lam = np.zeros((n, n_conf))
    for u in range(n_conf):
        ch = rng.choice(n, size=children, replace=False)
        Lam[ch, u] = rng.uniform(l_low, l_high, size=children) * rng.choice([-1, 1], size=children)
    return Lam


def norm_precision(Sigma):
    P = np.linalg.inv(Sigma)
    d = 1 / np.sqrt(np.diag(P))
    return d[:, None] * P * d[None, :]


def stats(W, Omega):
    n = W.shape[0]
    A = np.linalg.inv(np.eye(n) - W)
    Sigma = A @ Omega @ A.T
    P = norm_precision(Sigma)
    ev = np.linalg.eigvalsh(P)
    off = P[~np.eye(n, dtype=bool)]
    ev_om = np.linalg.eigvalsh(Omega)
    kappa = ev_om[-1] / ev_om[0]
    d_in = int((W != 0).sum(axis=1).max())
    bound = kappa * min(1 + d_in, np.linalg.norm(np.eye(n) - W, 2) ** 2)
    return ev[-1], ev[0], float(np.mean(off ** 2)), bound


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--p-edge", type=float, default=0.25)
    ap.add_argument("--n-conf", type=int, default=3)
    ap.add_argument("--children", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    print(f"n={a.n} p_edge={a.p_edge} confounders={a.n_conf} children={a.children} trials={a.trials} seed={a.seed}")
    for w_rng in [(0.1, 0.5), (0.5, 1.5)]:
        for l_rng in [(0.5, 1.0), (1.0, 2.0)]:
            rows = {"exp": [], "dd": []}
            for _ in range(a.trials):
                W = random_dag(rng, a.n, a.p_edge, *w_rng)
                Lam = random_loading(rng, a.n, a.n_conf, a.children, *l_rng)
                Om_exp = Lam @ Lam.T + np.eye(a.n)
                off = Om_exp - np.diag(np.diag(Om_exp))
                Om_dd = off + np.diag(1 + np.abs(off).sum(axis=1))
                rows["exp"].append(stats(W, Om_exp))
                rows["dd"].append(stats(W, Om_dd))
            e = np.array(rows["exp"]); d = np.array(rows["dd"])
            print(f"\n|W| in {w_rng}, |Lambda| in {l_rng}")
            for name, r in [("explicit", e), ("diag-dom", d)]:
                print(f"  {name:9s} rho(R~) median {np.median(r[:,0]):6.2f} max {r[:,0].max():7.2f} | "
                      f"min eig median {np.median(r[:,1]):.3f} | mean pcorr^2 {np.median(r[:,2]):.3f} | "
                      f"bound median {np.median(r[:,3]):7.2f} | bound holds {np.mean(r[:,0] <= r[:,3] + 1e-9):.0%}")
            print(f"  same instance: rho_exp > rho_dd in {np.mean(e[:,0] > d[:,0]):.0%};"
                  f" median ratio rho_exp/rho_dd {np.median(e[:,0] / d[:,0]):.2f};"
                  f" rho_exp above max rho_dd in {np.mean(e[:,0] > d[:,0].max()):.0%}")


if __name__ == "__main__":
    main()
