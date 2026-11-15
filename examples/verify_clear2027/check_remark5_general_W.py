"""Remark 5 for a general W: Omega = (1+c n)I - c 11^T (diagonally dominant); rho(R~) -> n as c grows when (I-W)^T 1 has no zero entry."""
import numpy as np
rng=np.random.default_rng(1)
n=8
for trial in range(4):
    W=np.zeros((n,n)); m=np.tril(rng.random((n,n))<0.3,-1); W[m]=rng.uniform(0.3,1.2,m.sum())*rng.choice([-1,1],m.sum())
    I=np.eye(n); u=(I-W).T@np.ones(n)
    out=[]
    for c in [1,10,100,1e4,1e6]:
        Om=(1+c*n)*I-c*np.ones((n,n)); Oi=np.linalg.inv(Om)
        Pr=(I-W).T@Oi@(I-W); d=np.sqrt(np.diag(Pr)); P=Pr/np.outer(d,d)
        out.append(round(float(np.linalg.eigvalsh(P)[-1]),3))
    print("min|u_i|=%.2f"%np.abs(u).min(), out)
