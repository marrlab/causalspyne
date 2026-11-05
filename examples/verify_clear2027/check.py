import numpy as np
np.set_printoptions(precision=4,suppress=True)
rho=lambda A:max(abs(np.linalg.eigvals(A)))
def pc(S):
    P=np.linalg.inv(S); d=np.sqrt(np.diag(P)); return P/np.outer(d,d)  # normalized precision (spectrum of R-tilde up to sign)
def Sig(W,O):
    A=np.linalg.inv(np.eye(len(W))-W); return A@O@A.T
print("A) Lemma 34 theta bound: one edge, Omega=I")
for w in [.5,1,2]:
    W=np.array([[0,0],[w,0.]]);S=Sig(W,np.eye(2))
    print(f" w={w}: rho(Sigma^-1)={rho(np.linalg.inv(S)):.4f}  claimed bound (rho(W+W^T)+1)*rho(Omega^-1)={rho(W+W.T)+1:.4f}")
print("B) Lemma 40: Sigma_ii <= 2 max Omega_ii")
for w in [.5,1,2]:
    W=np.array([[0,0],[w,0.]]);S=Sig(W,np.eye(2));print(f" w={w}: max Sigma_ii={S.diagonal().max():.3f} vs bound 2")
print("C) Theorem 41 random search for violations (DD Omega per Cor 19; Delta=1)")
rng=np.random.default_rng(0);viol=0;N=20000;worst=0
for _ in range(N):
    n=rng.integers(3,9);L=np.tril(rng.uniform(-.5,.5,(n,n))*(rng.random((n,n))<.4),-1)
    B=np.triu(rng.uniform(-.7,.7,(n,n))*(rng.random((n,n))<.3),1);O=B+B.T;O[np.diag_indices(n)]=1+abs(O).sum(1)
    S=Sig(L,O);d=((L!=0)|(L.T!=0)).sum(1).max();m=abs(L).max() if L.any() else 0
    b=2*O.diagonal().max()*(d*m+1)
    r=rho(pc(S))/b; worst=max(worst,r);viol+=r>1
print(f" violations {viol}/{N}, worst ratio actual/bound={worst:.3f}")
print("  tighter look at slack: median actual/bound over same type")
print("D) Theorem 41 with |w|=1..: chain")
for w in [1,3,10]:
    W=np.array([[0,0,0],[w,0,0],[0,w,0.]]);S=Sig(W,np.eye(3));b=2*(2*w+1)
    print(f" w={w}: rho(Rtilde)={rho(pc(S)):.3f}, Sigma_max={S.diagonal().max():.1f}, bound={b}")
print("E) Does W + diagonal (hence DD) Omega reach any PD Sigma? (Cholesky)")
n=30;u=np.ones(n);T=.95*np.ones((n,n))+.05*np.eye(n)   # equicorrelated, rho(R)=~|V|
C=np.linalg.cholesky(T);D=np.diag(np.diag(C));Lu=C@np.linalg.inv(D);W=np.eye(n)-np.linalg.inv(Lu);Om=D@D
print(f" recon err={abs(Sig(W,Om)-T).max():.2e}; rho(corr)={rho(T):.2f} of |V|={n}; Omega diagonal; max|W|={abs(W).max():.3f}")
T2=np.linalg.inv(np.eye(n)+.0); A=rng.normal(size=(n,3));T2=A@A.T+.01*np.eye(n);d=np.sqrt(T2.diagonal());T2=T2/np.outer(d,d)
print(f" rho(Rtilde) of low-rank-plus-noise target = {rho(pc(T2)):.2f}")
print("F) rho(R_eps)<=2 without identical diagonals; star under Cor 19")
for n,a in [(10,.5),(50,.5),(50,5)]:
    O=np.zeros((n,n));O[0,1:]=a;O[1:,0]=a;O[np.diag_indices(n)]=1+abs(O).sum(1);d=np.sqrt(O.diagonal());R=O/np.outer(d,d)
    off=abs(R[0]).sum()-1
    print(f" n={n},a={a}: row-0 off-diag sum of R={off:.3f} (DD needs<=1), rho(R)={rho(R):.3f}")
print("G) Lemma 27 strictness & Delta in Cor 19: Omega=I has lambda(Omega^-1)=1 =1/Delta (claim: strict <)")
print("H) entries vs radius: rho(Rtilde)>=1 always (trace=|V|); |entry|<=1 already")
print("I) Theorem 18 check: W1'' (21)/(31) vs true similarity")
nO,nU=4,2
Wo=np.tril(rng.normal(size=(nO,nO))*.5,-1);Lam=rng.normal(size=(nO,nU))
Wp=np.block([[Wo,Lam],[np.zeros((nU,nO)),np.zeros((nU,nU))]])
A=np.linalg.inv(np.eye(nO)-Wo);S11=A@(np.eye(nO)+Lam@Lam.T)@A.T;S12=A@Lam;S21=S12.T
Sp=np.linalg.inv(np.eye(nO+nU)-Wp)@np.eye(nO+nU)@np.linalg.inv(np.eye(nO+nU)-Wp).T
print(" Sigma' block OK:",abs(Sp[:nO,:nO]-S11).max()<1e-9,abs(Sp[:nO,nO:]-S12).max()<1e-9)
Si=np.linalg.inv(S11);T=S21@Si;Q=np.block([[np.eye(nO),Si@S12],[np.zeros((nU,nO)),np.eye(nU)]])
W1=np.block([[Wo,Lam-Wo@Si@S12],[T@Wo,T@(Lam-Wo@Si@S12)]])
print(" W1''(21) == Q^T W' Q^-1 :",np.allclose(W1,Q.T@Wp@np.linalg.inv(Q))," == Q^T W' Q^-T (true similarity):",np.allclose(W1,Q.T@Wp@np.linalg.inv(Q.T)))
Sn=Q.T@Sp@Q; eps=np.eye(nO+nU)
# structural identity Y''=W1''Y''+eps'' with Y''=Q^T Y', eps''=Q^T eps'
print(" (I-W1'')Q^T(I-W')^-1 == Q^T (structural eq holds):",np.allclose((np.eye(nO+nU)-W1)@Q.T@np.linalg.inv(np.eye(nO+nU)-Wp),Q.T))
print(" Implied Sigma_OO from (I-W1'')^-1 Q^T Q (I-W1'')^-T equals Sigma11:",np.allclose((np.linalg.inv(np.eye(nO+nU)-W1)@Q.T@Q@np.linalg.inv(np.eye(nO+nU)-W1).T)[:nO,:nO],S11))
print(" W1'' nilpotent(DAG-orderable)?",np.allclose(np.linalg.matrix_power(W1,nO+nU),0)," bottom-right block nonzero:",abs(T@(Lam-Wo@Si@S12)).max()>1e-9)
W2=np.block([[Wo+Lam@T,Lam],[np.zeros((nU,nO)),np.zeros((nU,nU))]])
Qi=np.linalg.inv(Q)
print(" W2''(31) == Q^-T W' Q^T:",np.allclose(W2,Qi.T@Wp@Q.T))
Y2=Qi.T; # Y''=Q^-T Y' (33)
print(" eq (31) structural: (I-W2'') Q^-T (I-W')^-1 maps eps' to noise; row2 =",np.round(((np.eye(nO+nU)-W2)@Qi.T@np.linalg.inv(np.eye(nO+nU)-Wp))[nO],3)[:6])
print("   eq (32) claims row2 = -S21 Si eps_O + xi:",np.round(np.r_[-T[0],1,0],3))
print("   cov(eps''_1,eps''_2) block zero?",np.allclose(((np.eye(nO+nU)-W2)@Qi.T@np.linalg.inv(np.eye(nO+nU)-Wp)@((np.eye(nO+nU)-W2)@Qi.T@np.linalg.inv(np.eye(nO+nU)-Wp)).T)[:nO,nO:],0))
print("J) non-root latent O1->U->{O2,O3} marginal in (W,Omega)")
a,b,c=1.2,.7,-.9
Wf=np.zeros((4,4));Wf[1,0]=a;Wf[2,1]=b;Wf[3,1]=c  # order O1,U,O2,O3
Sf=Sig(Wf,np.eye(4));keep=[0,2,3];So=Sf[np.ix_(keep,keep)]
Wo2=np.zeros((3,3));Wo2[1,0]=a*b;Wo2[2,0]=a*c;Om=np.eye(3);Om[1,1]=1+b*b;Om[2,2]=1+c*c;Om[1,2]=Om[2,1]=b*c
print(" marginal Sigma = (I-W)^-1 Omega (I-W)^-T with W, bidirected O2<->O3:",np.allclose(Sig(Wo2,Om),So)," Omega DD:",all(abs(Om[i,i])>=abs(Om[i]).sum()-abs(Om[i,i]) for i in range(3)))

print("K) Corrected Theorem: rho(Rt) <= kappa(Omega)*||I-W||_2^2 <= 2 maxOmega_ii (1+dmax max|W|)^2 max 1/Delta")
rng=np.random.default_rng(1);v1=v2=v0=0;N=20000;rs=[]
for _ in range(N):
    n=rng.integers(3,9);sc=rng.choice([.5,1,3,10]);L=np.tril(rng.uniform(-sc,sc,(n,n))*(rng.random((n,n))<.4),-1)
    B=np.triu(rng.uniform(-.7,.7,(n,n))*(rng.random((n,n))<.3),1);O=B+B.T;O[np.diag_indices(n)]=1+abs(O).sum(1)
    S=Sig(L,O);a=rho(pc(S));d=((L!=0)|(L.T!=0)).sum(1).max();m=abs(L).max() if L.any() else 0
    e=np.linalg.eigvalsh(O);b1=e[-1]/e[0]*np.linalg.norm(np.eye(n)-L,2)**2;b2=2*O.diagonal().max()*(1+d*m)**2
    b0=2*O.diagonal().max()*(1+d*m)
    v1+=a>b1*(1+1e-9);v2+=a>b2*(1+1e-9);v0+=a>b0*(1+1e-9);rs.append(a/b2)
print(f" original Thm41 violated {v0}/{N}; kappa-bound violated {v1}/{N}; squared-bound violated {v2}/{N}; median actual/squared-bound {np.median(rs):.3f}")
W=np.array([[0,0],[1.,0]]);
for w in [1,3,10,30]:
    W=np.array([[0,0,0],[w,0,0],[0,w,0.]]);print(f" chain w={w}: rho(Rt)={rho(pc(Sig(W,np.eye(3)))):.3f} original bound={2*(2*w+1)}")
print("L) avg squared partial correlation bound mean rho_ij^2 <= (rho-1)/(n-1)")
n=30;B=np.triu(rng.uniform(-.7,.7,(n,n))*(rng.random((n,n))<.1),1);O=B+B.T;O[np.diag_indices(n)]=1+abs(O).sum(1);L=np.tril(rng.uniform(-.5,.5,(n,n))*(rng.random((n,n))<.1),-1)
R=pc(Sig(L,O));off=R[~np.eye(n,dtype=bool)];r=rho(R);print(f" mean rho^2={np.mean(off**2):.4f} <= {(r-1)/(n-1):.4f}; max|rho_ij|={abs(off).max():.3f}; rho(Rt)={r:.3f}")
print("M) Remark 29: search lambda(Omega^-1)<B(V) with Omega positive entries, DD")
best=0
for _ in range(20000):
    n=rng.integers(3,8);l=rng.uniform(.01,1);A=np.triu(l+rng.exponential(rng.choice([.1,1,10]),(n,n)),1);O=A+A.T;O[np.diag_indices(n)]=O.sum(1)+rng.exponential(.01,n);lm=O[~np.eye(n,dtype=bool)].min()
    best=max(best,rho(np.linalg.inv(O))/((3*n-4)/(2*lm*(n-2)*(n-1))))
print(f" worst lambda/B(V)={best:.3f}")
print("N) Prop 74 star under Cor 19 with W2'' via Q^-1: ",end="");Qi=np.linalg.inv(Q);print(np.allclose(W2,Qi@Wp@Q.T))
Y=np.eye(nO+nU)
print(" eq (21)/(31) reproduce Sigma11 from structural form with their own noise? W1:",np.allclose((np.linalg.inv(np.eye(nO+nU)-W1)@(Q.T@Q)@np.linalg.inv(np.eye(nO+nU)-W1).T)[:nO,:nO],S11)," true W1=Q^T W' Q^-T:",np.allclose((np.linalg.inv(np.eye(nO+nU)-Q.T@Wp@np.linalg.inv(Q.T))@(Q.T@Q)@np.linalg.inv(np.eye(nO+nU)-Q.T@Wp@np.linalg.inv(Q.T)).T)[:nO,:nO],S11))
print(" W2:",np.allclose((np.linalg.inv(np.eye(nO+nU)-W2)@(Qi.T@Qi)@np.linalg.inv(np.eye(nO+nU)-W2).T)[:nO,:nO],S11))
