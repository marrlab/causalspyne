# Generates fig_admg_example.tex: DAG, ancestral ADMG, explicit block matrix W' and implied Omega for two examples.
pos = {1:(1.5,3), 2:(1.0,2), 5:(2.2,2), 6:(3.4,2), 0:(-0.6,1), 3:(2.2,1), 4:(1.5,0)}
dag = [(1,0),(1,2),(1,5),(2,0),(2,4),(5,3),(5,4),(6,3),(6,4),(0,4),(3,4)]  # parent -> child

def ancestral(hidden):
    import itertools
    anc = {v:set() for v in pos}
    changed = True
    while changed:
        changed = False
        for p,c in dag:
            new = {p} | anc[p]
            if not new <= anc[c]: anc[c] |= new; changed = True
    directed = {(p,c) for p,c in dag if p not in hidden and c not in hidden}
    bidir = set()
    for h in hidden:
        pa = [p for p,c in dag if c == h]; ch = [c for p,c in dag if p == h and c not in hidden]
        for p in pa:
            for c in ch:
                if p not in hidden: directed.add((p,c))
        for a,b in itertools.combinations(sorted(ch),2):
            if a in anc[b]: directed.add((a,b))
            elif b in anc[a]: directed.add((b,a))
            else: bidir.add((a,b))
    return directed, bidir

def graph(hidden, ancestral_flag):
    out = []
    directed, bidir = ancestral(hidden) if ancestral_flag else (set(dag), set())
    for v,(x,y) in pos.items():
        if ancestral_flag and v in hidden: continue
        st = 'hid' if v in hidden else 'obs'
        out.append(f"\\node[{st}] (n{v}) at ({x},{y}) {{{v}}};")
    for p,c in sorted(directed):
        bend = ',bend right=8' if (p,c)==(2,4) else (',bend left=24' if (p,c)==(6,4) else '')
        out.append(f"\\draw[arr{bend}] (n{p}) -- (n{c});")
    for a,b in sorted(bidir):
        bend = 'bend left=35' if abs(pos[a][1]-pos[b][1])<0.1 else 'bend left=20'
        out.append(f"\\draw[arr2,{bend}] (n{a}) to (n{b});")
    return "\n".join(out)

def matrix(obs, hid, entries, sym=False, labels=True, nm='M'):
    # rows: child, columns: parent. entries: set of (child, parent)
    order = list(obs) + list(hid)
    n, no = len(order), len(obs)
    s = 0.27
    out = []
    # block backgrounds
    def rect(r0,r1,c0,c1,col):
        out.append(f"\\fill[{col}] ({c0*s},{-r0*s}) rectangle ({c1*s},{-r1*s});")
    rect(0,no,0,no,'cO'); 
    if hid:
        rect(0,no,no,n,'cOU'); rect(no,n,0,no,'cUO'); rect(no,n,no,n,'cUU')
    for k in range(n+1):
        out.append(f"\\draw[gray!60,very thin] (0,{-k*s}) -- ({n*s},{-k*s}); \\draw[gray!60,very thin] ({k*s},0) -- ({k*s},{-n*s});")
    if hid:
        out.append(f"\\draw[thick] ({no*s},0) -- ({no*s},{-n*s}); \\draw[thick] (0,{-no*s}) -- ({n*s},{-no*s});")
    out.append(f"\\draw[thick] (0,0) rectangle ({n*s},{-n*s});")
    for i,r in enumerate(order):
        for j,c in enumerate(order):
            if (r,c) in entries:
                out.append(f"\\fill[black] ({(j+.5)*s},{-(i+.5)*s}) circle (0.075);")
    if labels:
        for j,c in enumerate(order):
            out.append(f"\\node[font=\\tiny,above=-1pt] at ({(j+.5)*s},0) {{{c}}};")
        for i,r in enumerate(order):
            out.append(f"\\node[font=\\tiny,left=-1pt] at (0,{-(i+.5)*s}) {{{r}}};")
    return "\n".join(out), n*s, no*s

def block_labels(no, n, s=0.27):
    return ""

def row(tag, hidden):
    obs = [v for v in sorted(pos) if v not in hidden]; hid = sorted(hidden)
    Wp = {(c,p) for p,c in dag}
    m1, w1, wo = matrix(obs, hid, Wp)
    directed, bidir = ancestral(hidden)
    Om = {(v,v) for v in obs} | {(a,b) for a,b in bidir} | {(b,a) for a,b in bidir}
    m2, w2, _ = matrix(obs, [], Om)
    lab = ""
    if hid:
        lab = (f"\\node[font=\\scriptsize,above=8pt] at ({wo/2},0) {{$W_O$}}; \\node[font=\\scriptsize,above=8pt] at ({(w1+wo)/2},0) {{$W_{{OU}}$}};"
               f"\\node[font=\\scriptsize,below=1pt] at ({wo/2},{-w1}) {{$W_{{UO}}$}}; \\node[font=\\scriptsize,below=1pt] at ({(w1+wo)/2},{-w1}) {{$W_{{UU}}$}};")
    return f"""
\\noindent\\textbf{{({tag})}}\\hfill
\\begin{{tikzpicture}}[x=0.68cm,y=0.72cm,baseline=(current bounding box.center)]
\\begin{{scope}}
\\node[font=\\scriptsize] at (1.4,3.8) {{DAG}};
{graph(hidden, False)}
\\end{{scope}}
\\begin{{scope}}[xshift=4.3cm]
\\node[font=\\scriptsize] at (1.4,3.8) {{ancestral ADMG}};
{graph(hidden, True)}
\\end{{scope}}
\\end{{tikzpicture}}\\hfill
\\begin{{tikzpicture}}[baseline=(current bounding box.center)]
\\node[font=\\scriptsize,anchor=south] at ({w1/2},0.75) {{explicit $W'$}};
{m1}
{lab}
\\end{{tikzpicture}}\\hfill
\\begin{{tikzpicture}}[baseline=(current bounding box.center)]
\\node[font=\\scriptsize,anchor=south] at ({w2/2},0.55) {{implicit $\\Omega$}};
{m2}
\\end{{tikzpicture}}
"""

tex = r"""% ADMG examples generated with causalspyne (main text, next to Example 1). Drawn in TikZ by gen.py (kept next to the repo).
% Same DAG in both rows. Row (a): hidden root vertex 1. Row (b): hidden vertices 2 and 5 (children of the observed vertex 1).
\claude{Figure 1 redrawn in TikZ on 2026-11-13 (the earlier matplotlib panels were small and low in resolution), with the explicit block matrix $W'$ and the implied $\Omega$ next to each graph. Both examples use the same 7-vertex DAG as the previous figure.}
\begin{figure}[t]
  \centering
  \definecolor{cO}{RGB}{214,230,250}\definecolor{cOU}{RGB}{252,226,196}\definecolor{cUO}{RGB}{213,238,214}\definecolor{cUU}{RGB}{232,232,232}
  \tikzset{obs/.style={circle,draw,thick,minimum size=0.45cm,inner sep=0pt,font=\scriptsize},
           hid/.style={circle,draw,thick,dashed,fill=gray!25,minimum size=0.45cm,inner sep=0pt,font=\scriptsize},
           arr/.style={-{Stealth[length=1.6mm]},thick},arr2/.style={{Stealth[length=1.6mm]}-{Stealth[length=1.6mm]},thick,red!70!black}}
""" + row("a", [1]) + "\n\\medskip\n" + row("b", [2,5]) + r"""
  \caption{Explicit unobserved confounding generated with causalspyne, the same DAG in both rows; hidden vertices are dashed. Each row shows the DAG, the ancestral ADMG over the observed vertices (red: bidirected edges), the weight matrix $W'$ of the explicit parameterization with blocks $W_O$, $W_{OU}$, $W_{UO}$, $W_{UU}$ (row $=$ child, column $=$ parent, observed vertices first) and the noise covariance $\Omega$ of the implicit parameterization, with a dot for every nonzero entry. (a) Hiding the root vertex~1 gives $W_{UO}=0$, $W_{UU}=0$ and $W_{OU}=\Lambda$ with three nonzero entries; its confounding appears as the bidirected edges $0\leftrightarrow5$ and $2\leftrightarrow5$, that is as off-diagonal entries of $\Omega$. (b) Hiding the vertices~2 and~5, which are children of the observed vertex~1, gives $W_{UO}\neq0$; no bidirected edge is created, so $\Omega$ is diagonal and the confounding is absorbed into directed edges out of vertex~1.}\label{fig:admg_example}
\end{figure}
"""
open('fig_admg_example_new.tex','w').write(tex)
print(ancestral([1]), ancestral([2,5]))
