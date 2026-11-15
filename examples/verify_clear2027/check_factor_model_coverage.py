"""Is every PD matrix of the form Lambda Lambda^T + D (D diagonal >0) with k factors? residual 0 means yes (numerical search over D).
Result: k = n-1 always; k = n-2 not always (n=4,k=2 and n=5,k=3 have PD matrices that are not reached)."""
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(0)
def covered(Om,k,tries=30):
    n=len(Om)
    # exists D>=0 diag with Om-D PSD and rank<=k  <=> sum of n-k smallest eigenvalues of Om-D = 0 with lambda_min>=0
    best=1e9
    for t in range(tries):
        d0=rng.uniform(0,np.linalg.eigvalsh(Om)[0],n)
        f=lambda d: np.sort(np.linalg.eigvalsh(Om-np.diag(d)))[:n-k].sum()
        cons=[{'type':'ineq','fun':lambda d: np.linalg.eigvalsh(Om-np.diag(d))[0]},{'type':'ineq','fun':lambda d: d}]
        r=minimize(f,d0,constraints=cons,method='SLSQP',options={'maxiter':300,'ftol':1e-12})
        if r.success or True:
            ok=np.linalg.eigvalsh(Om-np.diag(r.x))[0]>-1e-7
            if ok: best=min(best,f(r.x))
    return best
for n,k in [(3,1),(3,2),(4,1),(4,2),(4,3),(5,2),(5,3),(5,4)]:
    res=[]
    for _ in range(6):
        A=rng.normal(size=(n,n)); Om=A@A.T+0.1*np.eye(n)
        res.append(covered(Om,k))
    print(n,k,"min residual (0 => covered) over 6 random PD:",np.round(res,5))
