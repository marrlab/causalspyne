"""ICA-LiNGAM (lingam.ICALiNGAM; ICA-based, assumes no hidden confounder) under DD vs root-explicit confounding,
same fixed ADMG, same data pipeline as rcd_dd_vs_sdd.py (reused). Usage: python ica_dd_vs_sdd.py E|R reps [Ns]"""
import sys, numpy as np, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import rcd_dd_vs_sdd as r
import fci_dd_vs_pd as fd

def run_arm(W, pairs, Lam, Ue, UL, A, T):
    from lingam import ICALiNGAM
    Om = r.omega(Lam)
    X = (Ue + UL @ Lam.T) @ A.T
    X = (X - X.mean(0)) / X.std(0)
    m = ICALiNGAM(random_state=0); m.fit(X)
    return list(fd.spec(W, Om)) + r.rcd_metrics(W, pairs, m.adjacency_matrix_)

r.run_arm = run_arm

if __name__ == "__main__":
    from scipy.stats import wilcoxon
    design = sys.argv[1]; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    Ns = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [500, 2000]
    names = ["rho(R~)", "lmin(P)", "ICA SHD", "ICA dir R", "ICA dir P", "ICA <-> R", "ICA <-> P"]
    settings = [(a, b, c, N) for (a, b) in [(0.1, 0.5), (0.5, 1.5)] for c in [0.7, 2.0] for N in Ns]
    base = 40_000 if design == "E" else 50_000
    res = {}
    with Pool(16) as pool:
        for s, (wlo, whi, c, N) in enumerate(settings):
            R = np.array(pool.map(r.one, [(design, base * (s + 1) + k, 10, wlo, whi, c, N, 1500) for k in range(reps)]))
            key = f"W[{wlo},{whi}]_c{c}_N{N}"; res[key] = R
            print(f"{key}  design {design}, {len(R)} paired instances, means; Wilcoxon p")
            for q, nm in enumerate(names):
                dd, sd = R[:, 0, q], R[:, 1, q]
                p = wilcoxon(dd, sd).pvalue if np.any(dd != sd) else 1.0
                print(f"   {nm:10s} DD {dd.mean():.3f}  SDD {sd.mean():.3f}  p={p:.1e}", flush=True)
    np.savez(f"ica_{design}.npz", **res)
