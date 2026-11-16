"""Writes one non-Gaussian instance (same generator as rcd_dd_vs_sdd.py, design R, DD loadings) for ngBap: X.csv and truth.npz.
Usage: python ngbap_instance.py seed N outdir"""
import sys, os, numpy as np
import fci_rejection as fr, rcd_dd_vs_sdd as r
seed, N, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(seed)
n = 10; W, pairs = fr.instance(rng, n, 0.1, 0.5); k = len(pairs)
th = rng.uniform(-0.7, 0.7, k); Lam = r.loadings(th, np.zeros(k), pairs, n)
A = np.linalg.inv(np.eye(n) - W); s3 = np.sqrt(3)
X = (rng.uniform(-s3, s3, (N, n)) + rng.uniform(-s3, s3, (N, k)) @ Lam.T) @ A.T
X = (X - X.mean(0)) / X.std(0)
np.savetxt(os.path.join(out, "X.csv"), X, delimiter=","); np.savez(os.path.join(out, "truth.npz"), W=W, pairs=np.array(pairs))
print("pairs", pairs, "edges", int((W != 0).sum()))
