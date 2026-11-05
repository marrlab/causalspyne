"""Theorem 41 stress test + weight-free bound rho(Rt) <= kappa(Omega)*(1+max in-degree)."""
import numpy as np
rng=np.random.default_rng(7)
def stats(W,O):
    n=len(W);A=np.eye(n)-W;P=A.T@np.linalg.inv(O)@A;d=np.sqrt(np.diag(P));R=P/np.outer(d,d)
    r=np.linalg.eigvalsh((R+R.T)/2)[-1]
    sup=abs(W)>1e-12;dmax=(sup|sup.T).sum(1).max();din=sup.sum(1).max();m=abs(W).max() if sup.any() else 0
    De=np.diag(O)-(abs(O).sum(1)-abs(np.diag(O)));e=np.linalg.eigvalsh(O)
    return r,2*np.diag(O).max()*(dmax*m+1)/De.min(),e[-1]/e[0]*(1+din),e[-1]/e[0]*np.linalg.norm(A,2)**2,din
def ddO(n,dens,sc,slack=1.):
    B=np.triu(rng.uniform(-sc,sc,(n,n))*(rng.random((n,n))<dens),1);O=B+B.T;O[np.diag_indices(n)]=slack+abs(O).sum(1);return O
def rW(n,dens,sc):
    return np.tril(rng.uniform(-sc,sc,(n,n))*(rng.random((n,n))<dens),-1)
# (1) random sweep
N=30000;v41=vnew=vk=0;mx=0;mn=0
for _ in range(N):
    n=rng.integers(3,16);W=rW(n,rng.choice([.1,.3,.6,1]),rng.choice([.1,.5,1,3,10,100]));O=ddO(n,rng.choice([0,.1,.3,.8]),rng.choice([.1,.7,3]),rng.choice([.01,1,5]))
    r,b41,bn,bk,_=stats(W,O);v41+=r>b41*(1+1e-9);vnew+=r>bn*(1+1e-9);vk+=r>bk*(1+1e-9);mx=max(mx,r/b41);mn=max(mn,r/bn)
print(f"random {N}: Thm41 violations {v41}, kappa*(1+din) violations {vnew}, kappa*||I-W||^2 violations {vk}; worst r/b41={mx:.3f}, worst r/bnew={mn:.3f}")
# (2) hill climb directly on r/b41
best=0;bw=None
for rs in range(60):
    n=int(rng.integers(3,13));W=rW(n,.7,rng.choice([.1,.5,1.]));O=ddO(n,.3,.7,rng.choice([.01,1]))
    cur=stats(W,O);f=cur[0]/cur[1]
    for it in range(1500):
        W2=W*(1+rng.normal(0,.08,W.shape))+np.tril(rng.normal(0,.01,W.shape)*(rng.random(W.shape)<.1),-1)
        Bo=O.copy();np.fill_diagonal(Bo,0);Bo=Bo*(1+rng.normal(0,.08,Bo.shape));Bo=(Bo+Bo.T)/2;O2=Bo.copy();np.fill_diagonal(O2,rng.choice([.01,1])+abs(Bo).sum(1))
        c=stats(W2,O2);g=c[0]/c[1]
        if g>f:W,O,f=W2,O2,g
    if f>best:best=f;bw=(n,abs(W).max())
print(f"hill climb (60 restarts): best r/b41={best:.3f} at n,max|W|={bw}")
# (3) diagonal Omega=I: in-degree explains chain->2, weight scale irrelevant
print("weight scale sweep, Omega=I, chain (din=1) and 3-parent collider (din=3):")
for w in [.1,1,10,1000]:
    n=4;Wc=np.zeros((4,4));Wc[1,0]=Wc[2,1]=Wc[3,2]=w;Wk=np.zeros((4,4));Wk[3,:3]=w
    print(f"  w={w:<5}: chain rho={stats(Wc,np.eye(4))[0]:.3f} (<=2)   collider rho={stats(Wk,np.eye(4))[0]:.3f} (<=4)")
print("tightness: complete DAG, equal partial corr target gives rho close to 1+din:")
for n,c in [(10,.99),(30,.99)]:
    P=c*np.ones((n,n))+(1-c)*np.eye(n);C=np.linalg.cholesky(np.linalg.inv(P));D=np.diag(np.diag(C));W=np.eye(n)-np.linalg.inv(C@np.linalg.inv(D));O=D@D
    r=stats(W,O);print(f"  n={n}: rho={r[0]:.2f} din+1={r[4]+1} kappa={np.linalg.cond(O):.2f}")
# (4) median slack
o=[]
for _ in range(3000):
    n=20;W=rW(n,.15,.5);O=ddO(n,.1,.7);r=stats(W,O);o.append((r[0],r[1],r[2],r[3]))
o=np.median(np.array(o),0);print(f"typical n=20 sparse (W<=.5): rho={o[0]:.2f}; Thm41 RHS={o[1]:.1f}; kappa(1+din)={o[2]:.1f}; kappa||I-W||^2={o[3]:.1f}")
