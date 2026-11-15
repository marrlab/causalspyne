"""Numerical check of Lemma implicit_W_bows / Theorem 1(a): 500 random explicit models (8 vertices, 1-3 hidden vertices anywhere in the DAG).
Checks (i) Sigma_O = (I-Wimp)^-1 Omega (I-Wimp)^-T, (iii) support of Wimp, (iv) support of Omega, (v) ancestor pairs with Omega_ij!=0 are not ancestral. Result: 0 violations."""
import numpy as np, itertools
rng=np.random.default_rng(1)
bad=[0,0,0,0]; N=500
for t in range(N):
    n=8; nh=rng.integers(1,4)
    Wf=np.zeros((n,n)); m=np.tril(rng.random((n,n))<0.35,-1)
    Wf[m]=rng.uniform(0.3,1.2,m.sum())*rng.choice([-1,1],m.sum())
    perm=rng.permutation(n); Wf=Wf[np.ix_(perm,perm)]        # Wf[v,u]!=0: u->v ; vertices random order
    H=sorted(rng.choice(n,nh,replace=False).tolist()); O=[v for v in range(n) if v not in H]
    D=rng.uniform(0.5,2,n)                                   # noise variances
    I=np.eye(n); Sig=np.linalg.inv(I-Wf)@np.diag(D)@np.linalg.inv(I-Wf).T
    SigO=Sig[np.ix_(O,O)]
    WOO=Wf[np.ix_(O,O)]; WOU=Wf[np.ix_(O,H)]; WUO=Wf[np.ix_(H,O)]; WUU=Wf[np.ix_(H,H)]
    K=np.linalg.inv(np.eye(len(H))-WUU)
    Wimp=WOO+WOU@K@WUO
    Om=WOU@K@np.diag(D[H])@K.T@WOU.T+np.diag(D[O])
    S2=np.linalg.inv(np.eye(len(O))-Wimp)@Om@np.linalg.inv(np.eye(len(O))-Wimp).T
    if not np.allclose(S2,SigO,atol=1e-8): bad[0]+=1
    # support claims
    A=(Wf!=0).astype(int); R=A.copy()
    for _ in range(n): R=((R+R@A)>0).astype(int)          # R[v,u]=1: u ancestor of v
    Hs=set(H)
    # path o->...->o' with hidden interior (>=1 hidden), via DP over hidden-only walks
    def hidden_path(o,p):  # exists o->h1..->hk->p, hk,h1 hidden
        frontier=[h for h in H if Wf[h,o]!=0]; seen=set(frontier)
        while frontier:
            h=frontier.pop()
            if Wf[p,h]!=0: return True
            for h2 in H:
                if Wf[h2,h]!=0 and h2 not in seen: seen.add(h2); frontier.append(h2)
        return False
    for a,b in itertools.permutations(range(len(O)),2):
        want=WOO[b,a]!=0 or hidden_path(O[a],O[b])
        if want!=(abs(Wimp[b,a])>1e-12): bad[1]+=1
    def reach_hidden_int(u,p):  # path u->...->p with interior hidden (u hidden), length>=1
        if Wf[p,u]!=0: return True
        frontier=[h for h in H if Wf[h,u]!=0]; seen=set(frontier)
        while frontier:
            h=frontier.pop()
            if Wf[p,h]!=0: return True
            for h2 in H:
                if Wf[h2,h]!=0 and h2 not in seen: seen.add(h2); frontier.append(h2)
        return False
    for a,b in itertools.combinations(range(len(O)),2):
        want=any(reach_hidden_int(u,O[a]) and reach_hidden_int(u,O[b]) for u in H)
        if want!=(abs(Om[a,b])>1e-12): bad[2]+=1
        if want and (R[O[b],O[a]] or R[O[a],O[b]]):          # ancestor pair with nonzero Omega
            # then (Wimp,Omega) must not be ancestral: Wimp ancestors equal DAG ancestors on O
            A2=(np.abs(Wimp)>1e-12).astype(int); R2=A2.copy()
            for _ in range(len(O)): R2=((R2+R2@A2)>0).astype(int)
            if not (R2[b,a] or R2[a,b]): bad[3]+=1
print("instances",N,"violations: covariance",bad[0],"supp(Wimp)",bad[1],"supp(Omega)",bad[2],"ancestor-pair",bad[3])
