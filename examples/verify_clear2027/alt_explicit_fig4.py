"""Fig. 4 as an example of Theorem 1(b): for each panel build a second explicit parameterization (different graph / number of hidden vertices)
that gives the same implicit (W, Omega); checks equality numerically. Usage: python alt_explicit_fig4.py"""
import numpy as np, itertools
dag=[(1,0),(1,2),(1,5),(2,0),(2,4),(5,3),(5,4),(6,3),(6,4),(0,4),(3,4)]   # parent -> child (Fig. 4)
n=7; rng=np.random.default_rng(3)
def implicit(Wf,D,H):
    O=[v for v in range(n) if v not in H]
    WOO=Wf[np.ix_(O,O)];WOU=Wf[np.ix_(O,H)];WUO=Wf[np.ix_(H,O)];WUU=Wf[np.ix_(H,H)]
    K=np.linalg.inv(np.eye(len(H))-WUU)
    return O, WOO+WOU@K@WUO, WOU@K@np.diag(D[H])@K.T@WOU.T+np.diag(D[O])
def full(H):
    Wf=np.zeros((n,n))
    for p,c in dag: Wf[c,p]=rng.uniform(0.4,1.2)*rng.choice([-1,1])
    return Wf
for name,H in (("(a) hidden root 1",[1]),("(b) hidden 2,5",[2,5])):
    Wf=full(H); D=rng.uniform(1,2,n)
    O,Wi,Om=implicit(Wf,D,H)
    # alternative explicit model: root-explicit, observed block := Wi, one hidden root per bidirected pair of Omega
    k=len(O); pairs=[(a,b) for a,b in itertools.combinations(range(k),2) if abs(Om[a,b])>1e-12]
    Wf2=np.zeros((k+len(pairs),k+len(pairs))); Wf2[:k,:k]=Wi
    D2=np.r_[np.diag(Om).copy(), np.ones(len(pairs))]
    for q,(a,b) in enumerate(pairs):      # hidden root q loads on a (alpha) and b (beta), alpha*beta=Om_ab, equal magnitude
        al=np.sqrt(abs(Om[a,b])); be=np.sign(Om[a,b])*al
        Wf2[a,k+q]=al; Wf2[b,k+q]=be
        D2[a]-=al*al; D2[b]-=be*be
    ok=(D2[:k]>0).all()
    H2=list(range(k,k+len(pairs))); n_save=n
    n=k+len(pairs); O2,Wi2,Om2=implicit(Wf2,D2,H2); n=n_save
    print(name,"| observed",O,"| #hidden: original",len(H),"alternative",len(pairs),"| noise positive:",ok,
          "| same W_imp:",np.allclose(Wi,Wi2),"same Omega:",np.allclose(Om,Om2),"| W_OO original == W_imp?",np.allclose(Wf[np.ix_(O,O)],Wi))
    print("   bidirected pairs (observed labels):",[(O[a],O[b]) for a,b in pairs])
