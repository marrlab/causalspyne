"""Spectrum of the partial correlation matrix: ancestral (bow-free) vs bow-containing bidirected patterns.
Same fixed W (n=10, edge prob .3), same number k of bidirected pairs (each vertex in at most two pairs), same |Omega_ij|<=c.
  ancestral pattern: pairs (i,j) with neither vertex an ancestor of the other (the bidirected edges of an ancestral ADMG)
  bow pattern      : pairs (i,j) where i is an ancestor of j (these give bows or almost-directed cycles in (W, Omega))
For each pattern the largest rho(R~) over DD, SDD (root explicit) and PD Omega, with the search of fixedW_three_classes.py.
Usage: python bow_spectrum.py [--graphs 12] [--budget 600]"""
import argparse, numpy as np
import fixedW_extremes as fx
from fixedW_three_classes import search

def ancestors(W):
    n = len(W); A = (W != 0).astype(int); R = A.copy()   # R[j,i]=1: i ancestor of j
    for _ in range(n): R = ((R + R @ A) > 0).astype(int)
    return R

def pick(rng, cands, k, n):
    pairs, deg = [], np.zeros(n, int)
    for t in rng.permutation(len(cands)):
        i, j = cands[t]
        if deg[i] < 2 and deg[j] < 2:
            pairs.append((i, j)); deg[i] += 1; deg[j] += 1
        if len(pairs) == k: break
    return pairs

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--graphs", type=int, default=12)
    ap.add_argument("--budget", type=int, default=600); a = ap.parse_args()
    rng = np.random.default_rng(0); n = 10
    for wlo, whi in [(0.1, 0.5), (0.5, 1.5)]:
        for c in [0.7, 2.0]:
            rows = []
            while len(rows) < a.graphs:
                W = np.zeros((n, n)); m = np.tril(rng.random((n, n)) < 0.3, -1)
                W[m] = rng.uniform(wlo, whi, m.sum()) * rng.choice([-1, 1], m.sum())
                perm = rng.permutation(n); W = W[np.ix_(perm, perm)]
                R = ancestors(W)
                anc = [(i, j) for i in range(n) for j in range(i + 1, n) if not R[i, j] and not R[j, i]]
                bow = [(i, j) for i in range(n) for j in range(i + 1, n) if R[i, j] or R[j, i]]
                k = min(8, len(anc), len(bow))
                if k < 3: continue
                pa, pb = pick(rng, anc, k, n), pick(rng, bow, k, n)
                if len(pa) != len(pb) or not pa: continue
                row = []
                for pairs in (pa, pb):
                    for cls in ("DD", "SDD", "PD"):
                        row.append(search(rng, W, pairs, n, cls, "rho", c, a.budget))
                row[2], row[5] = max(row[2], row[1]), max(row[5], row[4])   # PD >= SDD
                rows.append(row)
            M = np.array(rows); med = np.median(M, 0)
            print(f"|W| in [{wlo},{whi}], c={c} ({len(rows)} W, medians): ancestral DD {med[0]:.2f} SDD {med[1]:.2f} PD {med[2]:.2f} | "
                  f"with bows DD {med[3]:.2f} SDD {med[4]:.2f} PD {med[5]:.2f} | median ratio bows/ancestral DD {np.median(M[:,3]/M[:,0]):.2f} PD {np.median(M[:,5]/M[:,2]):.2f}", flush=True)
