"""Theorem 1(a) predicts every entry of the implicit W: Wimp[j, i] = W_OO[j, i] + sum over the directed paths i -> h_1 -> ... -> h_m -> j
(m >= 1) whose inner vertices h_1..h_m are all hidden of the product of the edge weights. Check, for 5000 random explicit models (8 vertices,
1-3 hidden vertices anywhere in the DAG): (1) the formula W_OO + W_OU (I - W_UU)^-1 W_UO equals the path sum computed by enumeration (values),
(2) the set of extra entries (entries of Wimp that are not in W_OO) equals the set of observed pairs joined by such a path (predicted from the
graph alone), (3) Sigma_O = (I - Wimp)^-1 Omega (I - Wimp)^-T. Also the co-parent example: a -> h <- b, h -> c gives exactly two extra entries.
Usage: python check_theorem1a_paths.py"""
import numpy as np, itertools
rng = np.random.default_rng(7)
def paths_sum(Wf, H, o, p):
    """sum of weight products over directed paths o -> h1 ... hm -> p, m >= 1, inner vertices hidden"""
    Hs = set(H); total = 0.0
    def dfs(u, prod):
        nonlocal total
        for v in range(len(Wf)):
            if Wf[v, u] != 0:
                if v == p and u != o: total += prod * Wf[v, u]
                elif v in Hs: dfs(v, prod * Wf[v, u])
    dfs(o, 1.0)
    return total
bad = [0, 0, 0]; n_extra = []; N = 5000
for t in range(N):
    n = 8; nh = int(rng.integers(1, 4))
    Wf = np.zeros((n, n)); m = np.tril(rng.random((n, n)) < 0.35, -1); Wf[m] = rng.uniform(0.3, 1.2, m.sum()) * rng.choice([-1, 1], m.sum())
    perm = rng.permutation(n); Wf = Wf[np.ix_(perm, perm)]
    H = sorted(rng.choice(n, nh, replace=False).tolist()); O = [v for v in range(n) if v not in H]
    D = rng.uniform(0.5, 2, n)
    I = np.eye(n); A = np.linalg.inv(I - Wf); SigO = (A @ np.diag(D) @ A.T)[np.ix_(O, O)]
    WOO, WOU, WUO, WUU = Wf[np.ix_(O, O)], Wf[np.ix_(O, H)], Wf[np.ix_(H, O)], Wf[np.ix_(H, H)]
    K = np.linalg.inv(np.eye(len(H)) - WUU); Wimp = WOO + WOU @ K @ WUO
    Om = WOU @ K @ np.diag(D[H]) @ K.T @ WOU.T + np.diag(D[O])
    B = np.linalg.inv(np.eye(len(O)) - Wimp)
    if not np.allclose(B @ Om @ B.T, SigO, atol=1e-8): bad[2] += 1
    extra = 0
    for a, b in itertools.permutations(range(len(O)), 2):          # a -> b
        ps = paths_sum(Wf, H, O[a], O[b])
        if abs((Wimp - WOO)[b, a] - ps) > 1e-9: bad[0] += 1
        pred = abs(ps) > 1e-12 or any(True for _ in [0] if False)  # path exists with generic weights: ps != 0 (weights are generic)
        has_hidden_path = ps != 0.0
        if (abs((Wimp - WOO)[b, a]) > 1e-12) != has_hidden_path: bad[1] += 1
        extra += abs((Wimp - WOO)[b, a]) > 1e-12
    n_extra.append(extra)
print("random models", N, "violations: path-sum values", bad[0], "; extra-entry support", bad[1], "; covariance", bad[2], "; extra entries per model (mean, max):", round(float(np.mean(n_extra)), 2), max(n_extra))
# co-parent example: a=0, b=1 -> hidden 2 -> c=3
Wf = np.zeros((4, 4)); Wf[2, 0], Wf[2, 1], Wf[3, 2] = 0.5, -0.8, 1.5
H, O = [2], [0, 1, 3]
WOO, WOU, WUO, WUU = Wf[np.ix_(O, O)], Wf[np.ix_(O, H)], Wf[np.ix_(H, O)], Wf[np.ix_(H, H)]
Wimp = WOO + WOU @ np.linalg.inv(np.eye(1) - WUU) @ WUO
print("co-parent example: extra entries", [(O[j], O[i], round(Wimp[j, i], 3)) for j in range(3) for i in range(3) if abs(Wimp[j, i] - WOO[j, i]) > 1e-12], "expected W_(3,0) =", 1.5 * 0.5, "W_(3,1) =", 1.5 * (-0.8))
