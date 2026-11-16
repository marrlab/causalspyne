"""Is the ADMG used as ground truth in the SHD experiments (W, bidirected pairs) the same graph (adjacencies) as the exact MAG/PAG skeleton from dag2pag? Design E/R instances."""
import sys, numpy as np
sys.path.insert(0, ".")
import fci_rejection as fr, fci_dd_vs_pd as fd
bad = 0; tot = 0; extra = 0; miss = 0
for s in range(2000):
    rng = np.random.default_rng(1000 + s)
    for (wlo, whi) in [(0.1, 0.5), (0.5, 1.5)]:
        W, pairs = fr.instance(rng, 10, wlo, whi)
        n = 10
        T = fd.true_pag(W, pairs)
        adjT = (T != 0) & ~np.eye(n, dtype=bool)
        adjA = (W != 0) | (W.T != 0)
        for i, j in pairs: adjA[i, j] = adjA[j, i] = True
        tot += 1
        if (adjT != adjA).any():
            bad += 1; extra += int((adjT & ~adjA).sum() // 2); miss += int((adjA & ~adjT).sum() // 2)
print("instances", tot, "ADMG skeleton != exact PAG skeleton:", bad, "extra PAG edges", extra, "missing", miss)
