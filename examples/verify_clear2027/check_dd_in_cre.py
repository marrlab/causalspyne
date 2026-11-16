"""Numerical check: DD is a subset of CRE (the pair-root decomposition of a strictly diagonally dominant Omega reconstructs it exactly with positive D) and CRE is not a subset of DD
(about half of random CRE matrices with unequal loadings are not diagonally dominant). Usage: python check_dd_in_cre.py"""
import numpy as np
rng=np.random.default_rng(3)
# (a) DD subset of CRE: construct CRE parameters of random strictly DD Omega (pair roots, equal loadings, D=diag(Delta)) and reconstruct
maxerr=0; n_ok=0; N=20000
for t in range(N):
    n=int(rng.integers(3,11)); Om=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if rng.random()<0.4: Om[i,j]=Om[j,i]=rng.uniform(-1,1)
    delta=rng.uniform(0.01,1.0,n); np.fill_diagonal(Om,np.abs(Om-np.diag(np.diag(Om))).sum(1)+delta)   # strictly DD, margins delta
    pairs=[(i,j) for i in range(n) for j in range(i+1,n) if Om[i,j]!=0]
    Lam=np.zeros((n,len(pairs)))
    for l,(i,j) in enumerate(pairs):
        a=np.sqrt(abs(Om[i,j])); Lam[i,l]=a; Lam[j,l]=np.sign(Om[i,j])*a                    # equal loadings, two nonzero entries per column
    D=np.diag(delta)                                                                          # idiosyncratic variances, positive
    rec=Lam@Lam.T+D; err=np.abs(rec-Om).max(); maxerr=max(maxerr,err); n_ok+= (D.diagonal()>0).all() and ((Lam!=0).sum(0)==2).all()
print(f"(a) DD subset of CRE: {N} random strictly DD matrices (n=3..10, sparse random patterns): max reconstruction error {maxerr:.1e}; all have D>0 and exactly two nonzeros per column: {n_ok==N}")
# (b) CRE not subset of DD: sample CRE matrices with unequal loadings and count how many are not diagonally dominant
nonDD=0; M=20000
for t in range(M):
    n=int(rng.integers(3,9)); pairs=[(i,j) for i in range(n) for j in range(i+1,n) if rng.random()<0.5]
    if not pairs: continue
    Lam=np.zeros((n,len(pairs)))
    for l,(i,j) in enumerate(pairs):
        Lam[i,l]=rng.uniform(0.2,1.5)*rng.choice([-1,1]); Lam[j,l]=rng.uniform(0.2,1.5)*rng.choice([-1,1])
    Om=Lam@Lam.T+np.diag(rng.uniform(0.05,1,n)); off=np.abs(Om-np.diag(np.diag(Om))).sum(1)
    nonDD+= (np.diag(Om)<=off).any()
print(f"(b) CRE not subset of DD: {nonDD} of {M} random CRE matrices (unequal loadings, free signs) are NOT diagonally dominant ({100*nonDD/M:.1f}%)")
# example: 2 vertices... one pair a=1.5,b=0.2 : Omega_ii = a^2+d, Omega_ij=ab ; vertex j: b^2+d' vs ab
a,b,d=1.5,0.2,0.05; Om=np.array([[a*a+d,a*b],[a*b,b*b+d]]); print("explicit non-DD CRE example: Omega =",np.round(Om,3),"row 2: diag",round(Om[1,1],3),"< off-diagonal",round(Om[1,0],3))
