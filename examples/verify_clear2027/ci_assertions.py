"""CI assertions and Theorem 2 under DD vs the explicit extremes (PD), same W and pattern (design B instances).

For every pair (i,j) and conditioning set S, |S|<=2, the population partial correlation
pc(i,j|S) is computed exactly from Sigma = (I-W)^{-1} Omega (I-W)^{-T}.  FCI with Fisher-z rejects
'i independent of j given S' when |atanh(pc)| sqrt(N-|S|-3) > z_{0.975}.
Reported per setting (means over instances):
  same_zero    share of assertions that are true CI statements (pc=0) in BOTH arms and identical (must be 1: same ADMG)
  mean |pc|    over dependent assertions (pc != 0), DD vs PD
  flip@N       number of dependent assertions whose population test outcome differs between DD and PD
               (rejected in one arm, not in the other), at N=500 and 5000
  max flip example: the assertion with the largest |pc| difference
  Thm 2: rho(R~) / bound, bound = kappa(Omega) * min(1 + d_in_max, ||I-W||_2^2), max over instances (must be <= 1)
Usage: python ci_assertions.py [instances]
"""
import itertools, sys, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import faithfulness_margin as fm
import fci_dd_vs_pd as fd

SET = [(0.1, 0.5, 0.7), (0.1, 0.5, 2.0), (0.5, 1.5, 0.7), (0.5, 1.5, 2.0)]
Z = 1.959964


def pcs(S, n, smax=2):
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            rest = [k for k in range(n) if k not in (i, j)]
            for s in range(smax + 1):
                for cs in itertools.combinations(rest, s):
                    idx = [i, j, *cs]
                    P = np.linalg.inv(S[np.ix_(idx, idx)])
                    out[(i, j, cs)] = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
    return out


def one(a):
    s, r = a
    wlo, whi, c = SET[s]
    W, pairs, Oms = fm.inst_B(20_000 * (2 * s + 1) + r, 10, wlo, whi, c, 5000)  # same seeds as design B, N=5000 setting
    n = W.shape[0]
    A = np.linalg.inv(np.eye(n) - W)
    res = []
    for Om in Oms:
        Sg = A @ Om @ A.T
        pc = pcs(Sg, n)
        P = np.linalg.inv(Sg)
        d = np.sqrt(np.diag(P)); Rt = -P / np.outer(d, d); np.fill_diagonal(Rt, 1)
        rho = np.max(np.abs(np.linalg.eigvals(Rt)))
        ev = np.linalg.eigvalsh(Om)
        din = (np.abs(W) > 0).sum(1).max()
        bound = ev[-1] / ev[0] * min(1 + din, np.linalg.norm(np.eye(n) - W, 2) ** 2)
        res.append((pc, rho / bound))
    keys = list(res[0][0])
    a_, b_ = (np.array([res[k][0][q] for q in keys]) for k in (0, 1))
    zero_a, zero_b = np.abs(a_) < 1e-9, np.abs(b_) < 1e-9
    dep = ~zero_a & ~zero_b
    out = {"same_zero": float(np.mean(zero_a == zero_b)), "n_dep": int(dep.sum()),
           "mean_dd": float(np.abs(a_[dep]).mean()), "mean_pd": float(np.abs(b_[dep]).mean()),
           "thm2": (res[0][1], res[1][1])}
    for N in (500, 5000):
        z = lambda x, kk: np.abs(np.arctanh(np.clip(x, -0.999999, 0.999999))) * np.sqrt(N - np.array([len(q[2]) for q in kk]) - 3)
        za, zb = z(a_, keys), z(b_, keys)
        out[f"flip{N}"] = int(np.sum(dep & ((za > Z) != (zb > Z))))
        out[f"rej{N}"] = (int(np.sum(dep & (za > Z))), int(np.sum(dep & (zb > Z))))
    q = int(np.argmax(np.abs(np.abs(a_) - np.abs(b_)) * dep))
    out["ex"] = (keys[q], float(a_[q]), float(b_[q]))
    return out


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    with Pool(16) as pool:
        for s, (wlo, whi, c) in enumerate(SET):
            O = pool.map(one, [(s, r) for r in range(R)])
            m = lambda k: np.mean([o[k] for o in O])
            print(f"W[{wlo},{whi}] c={c} ({R} instances, |V|=10, |S|<=2)")
            print(f"   same CI zeros in both arms: {m('same_zero'):.3f}   dependent assertions/instance {m('n_dep'):.0f}")
            print(f"   mean |pc| over dependent: DD {m('mean_dd'):.3f}  PD {m('mean_pd'):.3f}")
            for N in (500, 5000):
                print(f"   N={N}: rejected DD {np.mean([o[f'rej{N}'][0] for o in O]):.1f} PD {np.mean([o[f'rej{N}'][1] for o in O]):.1f};"
                      f" outcome differs for {m(f'flip{N}'):.1f} assertions/instance ({100*m(f'flip{N}')/m('n_dep'):.1f}%)")
            ex = max(O, key=lambda o: abs(abs(o['ex'][1]) - abs(o['ex'][2])))['ex']
            print(f"   largest difference: pc({ex[0][0]},{ex[0][1]} | {list(ex[0][2])}) DD {ex[1]:+.3f}  PD {ex[2]:+.3f}")
            print(f"   Thm 2: max rho(R~)/bound  DD {max(o['thm2'][0] for o in O):.3f}  PD {max(o['thm2'][1] for o in O):.3f} (<=1 expected)", flush=True)
