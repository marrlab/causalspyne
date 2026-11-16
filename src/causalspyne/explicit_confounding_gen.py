"""Explicit-parameterization data generator of the CLeaR 2027 experiments (non-Gaussian data with hidden root confounders).

Model (root-explicit, Example 1 of the paper): X = (I - W)^{-1} (eps + Lambda xi), W strictly lower triangular up to a permutation,
eps and xi independent with i.i.d. uniform entries of unit variance, one hidden root xi_l per bidirected pair {i, j} of the ADMG
(canonical DAG), Lambda[:, l] nonzero only in rows i and j. Then Omega = I + Lambda Lambda^T (the covariance of the noise of the
implicit parameterization); with equal loadings a = b = sqrt(|theta|) it is the diagonally dominant Omega of the paper
(Omega_ii = 1 + sum_j |theta_ij|, Omega_ij = theta_ij); unequal loadings a_l / b_l = exp(2 r_l) give the canonical root-explicit class (CRE).
Data are standardized to unit column variance. Convention: W[child, parent] != 0 means parent -> child.
The functions use the same random number calls, in the same order, as the scripts in examples/verify_clear2027.
"""
import numpy as np


def random_instance(rng, n, w_low, w_high, p_edge=0.3, max_pairs_per_vertex=2):
    """Random weighted DAG W (lower triangular, edge probability p_edge, weights uniform in [w_low, w_high] with random sign, random
    vertex order) and the bidirected pairs: up to 4n random pairs i < j, each vertex in at most ``max_pairs_per_vertex`` pairs, and no
    pair that already has a directed edge (so the ADMG is bow-free). Returns (W, pairs)."""
    W = np.zeros((n, n))
    mask = np.tril(rng.random((n, n)) < p_edge, -1)
    W[mask] = rng.uniform(w_low, w_high, mask.sum()) * rng.choice([-1, 1], mask.sum())
    perm = rng.permutation(n)
    W = W[np.ix_(perm, perm)]
    pairs, deg = [], np.zeros(n, int)
    for _ in range(4 * n):
        i, j = sorted(rng.choice(n, 2, replace=False))
        if deg[i] < max_pairs_per_vertex and deg[j] < max_pairs_per_vertex and (i, j) not in pairs and W[i, j] == 0 and W[j, i] == 0:
            pairs.append((i, j))
            deg[i] += 1
            deg[j] += 1
    return W, pairs


def pair_loadings(theta, pairs, n, log_ratio=None):
    """Loading matrix Lambda (n x number of pairs): pair l = (i, j) gets a on i and sign(theta) * b on j with a b = |theta| and
    a / b = exp(2 r); r = 0 (equal loadings, diagonally dominant Omega) unless ``log_ratio`` is given."""
    k = len(pairs)
    r = np.zeros(k) if log_ratio is None else np.asarray(log_ratio, float)
    Lam = np.zeros((n, k))
    for l, ((i, j), t, rr) in enumerate(zip(pairs, theta, r)):
        a, b = np.sqrt(abs(t)) * np.exp(rr), np.sqrt(abs(t)) * np.exp(-rr)
        Lam[i, l], Lam[j, l] = a, np.sign(t) * b
    return Lam


def omega_from_loadings(Lam):
    """Omega = I + Lambda Lambda^T (unit idiosyncratic variances)."""
    return np.eye(Lam.shape[0]) + Lam @ Lam.T


def add_bows(rng, W, pairs, w_low, w_high):
    """Return a copy of W in which a random half (at least one) of the pairs gets a directed edge, oriented along a topological
    order, with weight uniform in [w_low, w_high] and random sign: these pairs become bows (confounded ancestor pairs)."""
    n = len(W)
    reach = (W.T != 0).astype(int)
    for _ in range(n):
        reach = ((reach + reach @ reach) > 0).astype(int)
    pos = np.argsort(np.argsort(reach.sum(0), kind="stable"), kind="stable")   # number of ancestors increases along every edge
    chosen = rng.choice(len(pairs), size=max(1, len(pairs) // 2), replace=False)
    W2 = W.copy()
    for l in chosen:
        i, j = pairs[l]
        a, b = (i, j) if pos[i] < pos[j] else (j, i)
        W2[b, a] = rng.uniform(w_low, w_high) * rng.choice([-1, 1])
    return W2


def sample_uniform_explicit(W, Lam, n_samples, rng, standardize=True):
    """n_samples rows of X = (I - W)^{-1} (eps + Lambda xi) with uniform unit-variance eps (n) and xi (number of pairs)."""
    n, k = Lam.shape
    s3 = np.sqrt(3)
    Ue = rng.uniform(-s3, s3, (n_samples, n))
    UL = rng.uniform(-s3, s3, (n_samples, k))
    X = (Ue + UL @ Lam.T) @ np.linalg.inv(np.eye(n) - W).T
    return (X - X.mean(0)) / X.std(0) if standardize else X
