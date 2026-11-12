"""For a FIXED DAG W, how far can Omega push the partial-correlation spectrum?

Classes of Omega (same confounded pairs, |Omega_ij| <= c, same noise floor):
  DD : diagonally dominant, Omega_ii = 1 + sum_j |Omega_ij|        (the paper's construction)
  PD : any positive definite Omega with the same off-diagonal pattern, lambda_min(Omega) >= 1
DD is a subset of PD (Gershgorin: lambda_min >= min margin = 1).
For each W we search (random starts + coordinate hill climbing, same budget per class) for
  max rho(R~), min lambda_min(P) (P = normalized precision, small = strong partial correlations),
  max mean squared partial correlation.
"""
import argparse
import numpy as np


def stats(W, Om):
    n = W.shape[0]
    A = np.eye(n) - W
    P = A.T @ np.linalg.solve(Om, A)
    d = 1 / np.sqrt(np.diag(P))
    P = d[:, None] * P * d[None, :]
    ev = np.linalg.eigvalsh(P)
    off = P[~np.eye(n, dtype=bool)]
    return ev[-1], ev[0], float(np.mean(off ** 2))


def build(theta, dvec, pairs, n, cls):
    O = np.zeros((n, n))
    for (i, j), t in zip(pairs, theta):
        O[i, j] = O[j, i] = t
    if cls == "DD":
        return O + np.diag(1 + np.abs(O).sum(1))
    O = O + np.diag(dvec)
    lmin = np.linalg.eigvalsh(O)[0]
    return O + max(0.0, 1.0 - lmin) * np.eye(n)


def objective(W, Om, which):
    r, lm, mp = stats(W, Om)
    return {"rho": r, "lmin": -lm, "pcorr2": mp}[which]


def search(rng, W, pairs, n, cls, which, c, budget):
    best, bestv = None, -np.inf
    k = len(pairs)
    for _ in range(budget // 4):
        th = rng.uniform(-c, c, k); dv = rng.uniform(0, 3 * c * 2, n)
        v = objective(W, build(th, dv, pairs, n, cls), which)
        if v > bestv: best, bestv = (th, dv), v
    th, dv = best[0].copy(), best[1].copy(); step = c / 2
    for it in range(budget):
        th2, dv2 = th.copy(), dv.copy()
        if k and rng.random() < 0.7:
            m = rng.integers(k); th2[m] = np.clip(th2[m] + rng.normal(0, step), -c, c)
        else:
            m = rng.integers(n); dv2[m] = max(0.0, dv2[m] + rng.normal(0, step))
        v = objective(W, build(th2, dv2, pairs, n, cls), which)
        if v > bestv: th, dv, bestv = th2, dv2, v
        if it % 300 == 299: step *= 0.7
    return bestv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--graphs", type=int, default=12)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n = a.n
    for wlo, whi in [(0.1, 0.5), (0.5, 1.5)]:
        for c in [0.7, 2.0]:
            rows = []
            for g in range(a.graphs):
                W = np.zeros((n, n)); m = np.tril(rng.random((n, n)) < 0.3, -1)
                W[m] = rng.uniform(wlo, whi, m.sum()) * rng.choice([-1, 1], m.sum())
                perm = rng.permutation(n); W = W[np.ix_(perm, perm)]
                # sparse confounding pattern: each variable in at most 2 pairs
                pairs, deg = [], np.zeros(n, int)
                for _ in range(4 * n):
                    i, j = sorted(rng.choice(n, 2, replace=False))
                    if deg[i] < 2 and deg[j] < 2 and (i, j) not in pairs:
                        pairs.append((i, j)); deg[i] += 1; deg[j] += 1
                base = stats(W, np.eye(n))
                r = [base[0], base[1], base[2]]
                for cls in ("DD", "PD"):
                    r.append(search(rng, W, pairs, n, cls, "rho", c, a.budget))
                    r.append(-search(rng, W, pairs, n, cls, "lmin", c, a.budget))
                    r.append(search(rng, W, pairs, n, cls, "pcorr2", c, a.budget))
                rows.append(r)
            R = np.median(np.array(rows), axis=0)
            print(f"|W| in [{wlo},{whi}], |Omega_ij|<={c}:  (medians over {a.graphs} fixed W, n={n})")
            print(f"   no confounding (Omega=I): rho {R[0]:.2f}  lmin {R[1]:.3f}  mean pcorr^2 {R[2]:.3f}")
            print(f"   DD extremes             : rho {R[3]:.2f}  lmin {R[4]:.3f}  mean pcorr^2 {R[5]:.3f}")
            print(f"   PD extremes (same pattern): rho {R[6]:.2f}  lmin {R[7]:.3f}  mean pcorr^2 {R[8]:.3f}", flush=True)


if __name__ == "__main__":
    main()
