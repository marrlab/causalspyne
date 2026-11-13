"""Fixed-W extremes for three nested Omega classes (same W, same bidirected pattern, |Omega_ij|<=c):
  DD  : Omega = I + sum_pairs |theta| [[1, s],[s, 1]]          (implicit, diagonally dominant)
  SDD : Omega = I + sum_pairs [[a^2, ab],[ab, b^2]], ab=theta   (explicit, root latents, one per
        pair, unequal loadings a=sqrt|theta|e^r, b=sqrt|theta|e^-r, |r|<=rmax; lower blocks zero)
  PD  : any PD with the pattern and lambda_min(Omega)>=1         (explicit, general; surjectivity)
        reported as max(PD search, SDD extreme), since SDD is a subset of PD.
DD (r=0) is a subset of SDD; SDD is a subset of PD. Same search budget per class."""
import argparse, numpy as np
import fixedW_extremes as fx

def build(x, pairs, n, cls, c):
    k = len(pairs)
    if cls == "PD":
        return fx.build(x[:k], x[k:k + n], pairs, n, "PD")
    th, r = x[:k], (x[k:2 * k] if cls == "SDD" else np.zeros(k))
    O = np.eye(n)
    for (i, j), t, rr in zip(pairs, th, r):
        a, b = np.sqrt(abs(t)) * np.exp(rr), np.sqrt(abs(t)) * np.exp(-rr)
        O[i, i] += a * a; O[j, j] += b * b; O[i, j] += np.sign(t) * a * b; O[j, i] = O[i, j]
    return O

def search(rng, W, pairs, n, cls, which, c, budget, rmax=2.0):
    k = len(pairs)
    dim = k + (n if cls == "PD" else k if cls == "SDD" else 0)
    lo = np.r_[-c * np.ones(k), (0 if cls == "PD" else -rmax) * np.ones(dim - k)]
    hi = np.r_[c * np.ones(k), (np.exp(2 * rmax) * c if cls == "PD" else rmax) * np.ones(dim - k)]
    f = lambda x: fx.objective(W, build(x, pairs, n, cls, c), which)
    best = max((rng.uniform(lo, hi) for _ in range(budget // 4)), key=f)
    bv, step = f(best), c / 2
    for it in range(budget):
        x = best.copy(); m = rng.integers(dim)
        x[m] = np.clip(x[m] + rng.normal(0, step), lo[m], hi[m])
        if cls == "PD" and m >= k: x[m] = max(0.0, x[m])
        v = f(x)
        if v > bv: best, bv = x, v
        if it % 300 == 299: step *= 0.7
    return bv

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--graphs", type=int, default=24)
    ap.add_argument("--budget", type=int, default=1500); a = ap.parse_args()
    rng = np.random.default_rng(0); n = 10
    for wlo, whi in [(0.1, 0.5), (0.5, 1.5)]:
        for c in [0.7, 2.0]:
            rows = []
            for g in range(a.graphs):
                W = np.zeros((n, n)); m = np.tril(rng.random((n, n)) < 0.3, -1)
                W[m] = rng.uniform(wlo, whi, m.sum()) * rng.choice([-1, 1], m.sum())
                perm = rng.permutation(n); W = W[np.ix_(perm, perm)]
                pairs, deg = [], np.zeros(n, int)
                for _ in range(4 * n):
                    i, j = sorted(rng.choice(n, 2, replace=False))
                    if deg[i] < 2 and deg[j] < 2 and (i, j) not in pairs:
                        pairs.append((i, j)); deg[i] += 1; deg[j] += 1
                r = list(fx.stats(W, np.eye(n)))
                for cls in ("DD", "SDD", "PD"):
                    r += [search(rng, W, pairs, n, cls, "rho", c, a.budget),
                          -search(rng, W, pairs, n, cls, "lmin", c, a.budget),
                          search(rng, W, pairs, n, cls, "pcorr2", c, a.budget)]
                # SDD is a subset of PD: the PD extreme is at least the SDD extreme
                r[9], r[11] = max(r[9], r[6]), max(r[11], r[8]); r[10] = min(r[10], r[7])
                rows.append(r)
            R = np.median(np.array(rows), 0)
            print(f"|W| in [{wlo},{whi}], c={c} (medians over {a.graphs} W): "
                  + "  ".join(f"{cl}: rho {R[3*q]:.2f} lmin {R[3*q+1]:.4f} pc2 {R[3*q+2]:.3f}"
                              for q, cl in enumerate(("none", "DD", "SDD", "PD"))), flush=True)
