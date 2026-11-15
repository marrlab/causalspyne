"""Is DAG2Ancestral (Algorithm 1 of the paper) the MAG of the DAG over the observed variables?

Compares its output with the exact MAG (edge a-b iff no set of observed variables d-separates a and b in the DAG;
mark a->b if a is an ancestor of b, a<->b otherwise). Usage: python check_dag2ancestral_vs_mag.py K   (K = number of hidden nodes)
Result (400 random DAGs, 7 nodes, edge prob .35): never a bow, never a wrong or extra edge, but missing edges in
17/400 (K=1), 55/400 (K=2), 54/400 (K=3) instances; minimal counterexample: a->c, h->c, h->b, c->b with h hidden (misses a->b)."""
import itertools, numpy as np, networkx as nx, sys
from causalspyne.dag2ancestral import DAG2Ancestral
rng = np.random.default_rng(0)
def true_mag(adj, hidden):
    n = len(adj); G = nx.DiGraph([(p, c) for c in range(n) for p in range(n) if adj[c, p]]); G.add_nodes_from(range(n))
    O = [v for v in range(n) if v not in hidden]
    an = {v: nx.ancestors(G, v) for v in range(n)}
    dsep = nx.is_d_separator if hasattr(nx, "is_d_separator") else nx.d_separated
    edges = {}
    for a, b in itertools.combinations(O, 2):
        rest = [v for v in O if v not in (a, b)]
        adjacent = True
        for k in range(len(rest) + 1):
            for S in itertools.combinations(rest, k):
                if dsep(G, {a}, {b}, set(S)): adjacent = False; break
            if not adjacent: break
        if adjacent:
            if a in an[b]: edges[(a, b)] = "->"
            elif b in an[a]: edges[(b, a)] = "->"
            else: edges[(a, b)] = "<->"
    return edges
def algo_edges(adj, hidden):
    A = DAG2Ancestral(adj).run(list(hidden)); O = [v for v in range(len(adj)) if v not in hidden]
    edges = {}
    for i, a in enumerate(O):
        for j, b in enumerate(O):
            if i < j:
                if A[j, i] and A[i, j]: edges[(a, b)] = "<->"
                elif A[j, i]: edges[(a, b)] = "->"
                elif A[i, j]: edges[(b, a)] = "->"
    return edges
import sys as _s
K=int(_s.argv[1])
n = 7; bad = 0; tot = 0; bows = 0; kinds = {"extra": 0, "missing": 0, "type": 0}
ex = []
for t in range(400):
    adj = np.zeros((n, n), int)
    for j in range(n):
        for i in range(j):
            if rng.random() < 0.35: adj[j, i] = 1   # i -> j (W convention W[child,parent])
    hidden = sorted(rng.choice(n, size=K, replace=False).tolist())
    T = true_mag(adj, hidden); A = algo_edges(adj, hidden); tot += 1
    # bows in algo output
    if any(((b, a) in A and A[(b, a)] == "->" or (a, b) in A and A[(a, b)] == "->") and t == "<->" for (a, b), t in A.items()): bows += 1
    if T != A:
        bad += 1
        for k in set(T) | set(A):
            if k not in A: kinds["missing"] += 1
            elif k not in T: kinds["extra"] += 1
            elif T[k] != A[k]: kinds["type"] += 1
        if len(ex) < 2: ex.append((adj.tolist(), hidden, T, A))
print("instances", tot, "algo != true MAG:", bad, kinds, "algo outputs with bows:", bows)
if ex: print(ex[0][1], ex[0][2], ex[0][3])
