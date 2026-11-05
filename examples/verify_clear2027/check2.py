import numpy as np
rho=lambda A:max(abs(np.linalg.eigvalsh((A+A.T)/2)))
def pc(S):
    P=np.linalg.inv(S); d=np.sqrt(np.diag(P)); return P/np.outer(d,d)
for n in [5,10,30,100]:
  for c in [.5,.9,.99]:
    P=c*np.ones((n,n))+(1-c)*np.eye(n)          # target precision -> equal partial corr, rho(Rt)=1+(n-1)c
    T=np.linalg.inv(P);C=np.linalg.cholesky(T);D=np.diag(np.diag(C));Lu=C@np.linalg.inv(D);W=np.eye(n)-np.linalg.inv(Lu);Om=D@D
    A=np.linalg.inv(np.eye(n)-W);err=abs(A@Om@A.T-T).max();d=((abs(W)>1e-12)|(abs(W.T)>1e-12)).sum(1).max();m=abs(W).max()
    r=rho(pc(T));Delta=np.diag(Om);b=2*Delta.max()*(d*m+1)/Delta.min()
    print(f"n={n:3d} c={c}: rho(Rt)={r:7.2f}/|V|={n}; Omega diagonal; max|W|={m:6.3f}, dmax={d}; RHS Thm41={b:10.2f} {'VIOLATED' if r>b else 'ok'} err={err:.1e}")
# scale Omega so Delta-independent: normalized comparison, W bounded by .5 ?
print('--- same but constrained |W|<=0.5, diagonal Omega, sparse DAG: random search for max rho(Rt)/|V|')
rng=np.random.default_rng(0)
for n in [10,30]:
  best=0
  for _ in range(3000):
    p=rng.choice([.1,.3,.6,1]);L=np.tril(rng.uniform(-.5,.5,(n,n))*(rng.random((n,n))<p),-1);S=np.linalg.inv(np.eye(n)-L);S=S@S.T
    best=max(best,rho(pc(S)))
  print(f" n={n}: max rho(Rt)={best:.2f}")
print('--- DD vs non-DD Omega under the SAME W: rho(Rt) of Omega=random PD with same support vs DD, n=20')
n=20;out=[]
for _ in range(300):
    L=np.tril(rng.uniform(-.5,.5,(n,n))*(rng.random((n,n))<.2),-1);A=np.linalg.inv(np.eye(n)-L)
    B=np.triu(rng.uniform(-.7,.7,(n,n))*(rng.random((n,n))<.15),1);Od=B+B.T;Od[np.diag_indices(n)]=1+abs(Od).sum(1)
    F=rng.normal(size=(n,3));On=F@F.T+.05*np.eye(n)
    out.append((rho(pc(A@Od@A.T)),rho(pc(A@On@A.T))))
o=np.array(out);print(f" DD median rho(Rt)={np.median(o[:,0]):.2f}; factor-model PD Omega median={np.median(o[:,1]):.2f}; max {o[:,0].max():.2f} vs {o[:,1].max():.2f}")
