"""
Illustrate hidden confounding for a DAG: the DAG, its ancestral ADMG over the
observed vertices, the explicit block weight matrix W' and the implied noise
covariance pattern Omega, in one figure that can be written as a Matplotlib
figure or as TikZ code for a LaTeX paper.

Convention (the CausalSpyne one): ``adj[i, j] != 0`` means an edge from j to i,
so a row is a child and a column is a parent.

The explicit parameterization of unobserved confounding orders the vertices
observed first, hidden last::

    W' = [[W_O , W_OU],
          [W_UO, W_UU]]

For hidden root vertices W_UO = 0 and W_UU = 0. The implicit parameterization
keeps only the observed vertices; its Omega has a nonzero off-diagonal entry
for every bidirected edge of the ancestral ADMG, which is computed by
:class:`causalspyne.dag2ancestral.DAG2Ancestral`.

Example::

    ill = ConfoundingIllustration(adj, hidden=[1])
    ill.to_matplotlib().savefig("fig.pdf")
    open("fig.tex", "w").write(ill.to_tikz())
"""
from __future__ import annotations

import numpy as np

from causalspyne.dag2ancestral import DAG2Ancestral

# light block colours: W_O, W_OU, W_UO, W_UU
_BLOCK_RGB = {
    "O": (214, 230, 250),
    "OU": (252, 226, 196),
    "UO": (213, 238, 214),
    "UU": (232, 232, 232),
}


def layered_positions(adj, sweeps=6):
    """
    Layered layout. The layer of a vertex is the length of the longest directed
    path into it; the order inside a layer is improved by a few barycenter
    sweeps (mean position of the neighbours) to reduce edges that pass behind
    other vertices. Returns {vertex: (x, y)} with y growing upward.
    """
    adj = np.asarray(adj)
    n = adj.shape[0]
    depth = np.zeros(n, dtype=int)
    for _ in range(n):  # longest path by relaxation, terminates for a DAG
        for child, parent in zip(*np.nonzero(adj)):
            depth[child] = max(depth[child], depth[parent] + 1)
    max_depth = int(depth.max()) if n else 0
    layers = [[v for v in range(n) if depth[v] == k] for k in range(max_depth + 1)]
    xpos = {}
    for layer in layers:
        for k, v in enumerate(layer):
            xpos[v] = k - (len(layer) - 1) / 2
    neighbours = {v: [int(u) for u in np.nonzero(adj[v])[0]]
                  + [int(u) for u in np.nonzero(adj[:, v])[0]] for v in range(n)}
    for sweep in range(sweeps):
        order = range(len(layers)) if sweep % 2 == 0 else reversed(range(len(layers)))
        for k in order:
            layer = layers[k]
            bary = {v: (np.mean([xpos[u] for u in neighbours[v]])
                        if neighbours[v] else xpos[v]) for v in layer}
            layer.sort(key=lambda v: (bary[v], v))
            for r, v in enumerate(layer):
                xpos[v] = r - (len(layer) - 1) / 2
    width = max(len(layer) for layer in layers) if layers else 1
    return {v: (xpos[v] * 1.25 + (width - 1) * 0.625, float(max_depth - depth[v]))
            for v in range(n)}


class ConfoundingIllustration:
    """
    Parameters
    ----------
    adj : (n, n) array, ``adj[i, j] != 0`` iff j -> i (acyclic)
    hidden : global indices of the hidden vertices
    pos : optional {vertex: (x, y)}; default is :func:`layered_positions`
    """

    def __init__(self, adj, hidden, pos=None):
        self.adj = (np.asarray(adj) != 0).astype(int)
        self.n = self.adj.shape[0]
        self.hidden = sorted(int(h) for h in hidden)
        self.observed = [v for v in range(self.n) if v not in self.hidden]
        self.pos = pos if pos is not None else layered_positions(self.adj)
        anc = DAG2Ancestral(self.adj).run(self.hidden)  # observed x observed
        obs = self.observed
        self.directed = []  # (parent, child) in global indices
        self.bidirected = []  # (a, b), a < b
        for i, ci in enumerate(obs):
            for j, cj in enumerate(obs):
                if i == j or not anc[i, j]:
                    continue
                if anc[j, i]:
                    if ci < cj:
                        self.bidirected.append((ci, cj))
                else:
                    self.directed.append((cj, ci))  # anc[i, j]: cj -> ci

    # ------------------------------------------------------------------ matrices
    @property
    def order(self):
        """vertex order of the explicit matrix: observed first, hidden last"""
        return self.observed + self.hidden

    def explicit_matrix(self):
        """W' of the explicit parameterization, observed vertices first"""
        idx = np.array(self.order)
        return self.adj[np.ix_(idx, idx)]

    def omega_pattern(self):
        """nonzero pattern of the implied Omega over the observed vertices"""
        obs = self.observed
        pos = {v: k for k, v in enumerate(obs)}
        pattern = np.eye(len(obs), dtype=int)
        for a, b in self.bidirected:
            pattern[pos[a], pos[b]] = pattern[pos[b], pos[a]] = 1
        return pattern

    # ------------------------------------------------------------- matplotlib
    def to_matplotlib(self, figsize=(11, 3.4)):
        """one row: DAG | ancestral ADMG | explicit W' | implicit Omega"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, axes = plt.subplots(
            1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1.3, 1.3, 1, 1]}
        )
        self._draw_graph(axes[0], full=True)
        axes[0].set_title("DAG")
        self._draw_graph(axes[1], full=False)
        axes[1].set_title("ancestral ADMG")

        def draw_matrix(ax, mat, labels, n_obs, title):
            n = len(labels)
            if n_obs < n:
                blocks = [(0, n_obs, 0, n_obs, "O"), (0, n_obs, n_obs, n, "OU"),
                          (n_obs, n, 0, n_obs, "UO"), (n_obs, n, n_obs, n, "UU")]
            else:
                blocks = [(0, n, 0, n, "O")]
            for r0, r1, c0, c1, kind in blocks:
                colour = tuple(c / 255 for c in _BLOCK_RGB[kind])
                ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,
                                       facecolor=colour, edgecolor="none"))
            for i, j in zip(*np.nonzero(mat)):
                ax.plot(j + 0.5, i + 0.5, "ko", ms=5)
            for k in range(n + 1):
                ax.axhline(k, color="0.6", lw=0.4)
                ax.axvline(k, color="0.6", lw=0.4)
            if n_obs < n:
                ax.axhline(n_obs, color="k", lw=1.4)
                ax.axvline(n_obs, color="k", lw=1.4)
            ax.set_xlim(0, n)
            ax.set_ylim(n, 0)
            ax.set_aspect("equal")
            ax.set_xticks(np.arange(n) + 0.5)
            ax.set_xticklabels(labels, fontsize=7)
            ax.xaxis.tick_top()
            ax.set_yticks(np.arange(n) + 0.5)
            ax.set_yticklabels(labels, fontsize=7)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
            ax.set_title(title, pad=22)

        draw_matrix(axes[2], self.explicit_matrix(), self.order, len(self.observed),
                    "explicit $W'$")
        draw_matrix(axes[3], self.omega_pattern(), self.observed, len(self.observed),
                    r"implicit $\Omega$")
        fig.tight_layout()
        return fig

    def _draw_graph(self, ax, full):
        hidden = set(self.hidden)
        directed = ([(p, c) for c, p in zip(*np.nonzero(self.adj))]
                    if full else self.directed)
        nodes = [v for v in range(self.n) if full or v not in hidden]
        for p, c in directed:
            ax.annotate("", xy=self.pos[c], xytext=self.pos[p],
                        arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "k",
                                    "shrinkA": 11, "shrinkB": 11})
        if not full:
            for a, b in self.bidirected:
                ax.annotate("", xy=self.pos[b], xytext=self.pos[a],
                            arrowprops={"arrowstyle": "<|-|>", "lw": 1.1,
                                        "color": "firebrick", "shrinkA": 11,
                                        "shrinkB": 11, "connectionstyle": "arc3,rad=0.3"})
        for v in nodes:
            x, y = self.pos[v]
            hid = v in hidden
            ax.scatter([x], [y], s=380, facecolor="0.85" if hid else "white",
                       edgecolor="k", linestyle="--" if hid else "-", zorder=3)
            ax.text(x, y, str(v), ha="center", va="center", fontsize=8, zorder=4)
        xs = [p[0] for p in self.pos.values()]
        ys = [p[1] for p in self.pos.values()]
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
        ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
        ax.axis("off")

    # -------------------------------------------------------------------- TikZ
    def _tikz_graph(self, full, xscale, yscale):
        hidden = set(self.hidden)
        lines = []
        for v in range(self.n):
            if not full and v in hidden:
                continue
            x, y = self.pos[v]
            style = "hid" if v in hidden else "obs"
            lines.append(f"\\node[{style}] (n{v}) at ({x:.2f},{y:.2f}) {{{v}}};")
        directed = ([(p, c) for c, p in zip(*np.nonzero(self.adj))]
                    if full else self.directed)
        for p, c in sorted(directed):
            lines.append(f"\\draw[arr] (n{p}) -- (n{c});")
        if not full:
            for a, b in self.bidirected:
                bend = "bend left=35" if abs(self.pos[a][1] - self.pos[b][1]) < 0.1 \
                    else "bend left=20"
                lines.append(f"\\draw[arr2,{bend}] (n{a}) to (n{b});")
        return "\n".join(lines)

    def _tikz_matrix(self, mat, labels, n_obs, title, cell=0.27):
        n = len(labels)
        out = [f"\\node[font=\\scriptsize,anchor=south] at ({n * cell / 2:.3f},0.75) {{{title}}};"]

        def rect(r0, r1, c0, c1, colour):
            out.append(f"\\fill[{colour}] ({c0 * cell:.3f},{-r0 * cell:.3f}) "
                       f"rectangle ({c1 * cell:.3f},{-r1 * cell:.3f});")

        if n_obs < n:
            rect(0, n_obs, 0, n_obs, "cO"); rect(0, n_obs, n_obs, n, "cOU")
            rect(n_obs, n, 0, n_obs, "cUO"); rect(n_obs, n, n_obs, n, "cUU")
        else:
            rect(0, n, 0, n, "cO")
        for k in range(n + 1):
            out.append(f"\\draw[gray!60,very thin] (0,{-k * cell:.3f}) -- ({n * cell:.3f},{-k * cell:.3f});"
                       f" \\draw[gray!60,very thin] ({k * cell:.3f},0) -- ({k * cell:.3f},{-n * cell:.3f});")
        if n_obs < n:
            out.append(f"\\draw[thick] ({n_obs * cell:.3f},0) -- ({n_obs * cell:.3f},{-n * cell:.3f});"
                       f" \\draw[thick] (0,{-n_obs * cell:.3f}) -- ({n * cell:.3f},{-n_obs * cell:.3f});")
        out.append(f"\\draw[thick] (0,0) rectangle ({n * cell:.3f},{-n * cell:.3f});")
        for i, j in zip(*np.nonzero(mat)):
            out.append(f"\\fill ({(j + .5) * cell:.3f},{-(i + .5) * cell:.3f}) circle (0.075);")
        for j, v in enumerate(labels):
            out.append(f"\\node[font=\\tiny,above=-1pt] at ({(j + .5) * cell:.3f},0) {{{v}}};")
        for i, v in enumerate(labels):
            out.append(f"\\node[font=\\tiny,left=-1pt] at (0,{-(i + .5) * cell:.3f}) {{{v}}};")
        if n_obs < n:
            w, wo = n * cell, n_obs * cell
            out.append(f"\\node[font=\\scriptsize,above=8pt] at ({wo / 2:.3f},0) {{$W_O$}};"
                       f" \\node[font=\\scriptsize,above=8pt] at ({(w + wo) / 2:.3f},0) {{$W_{{OU}}$}};"
                       f" \\node[font=\\scriptsize,below=1pt] at ({wo / 2:.3f},{-w:.3f}) {{$W_{{UO}}$}};"
                       f" \\node[font=\\scriptsize,below=1pt] at ({(w + wo) / 2:.3f},{-w:.3f}) {{$W_{{UU}}$}};")
        return "\n".join(out)

    def tikz_preamble(self):
        """definitions needed once per document (colours and node styles); the
        packages ``tikz`` with libraries ``arrows.meta`` and ``bending`` are required"""
        colours = "".join(
            f"\\definecolor{{c{k}}}{{RGB}}{{{r},{g},{b}}}"
            for k, (r, g, b) in _BLOCK_RGB.items())
        return (colours + "\n"
                "\\tikzset{obs/.style={circle,draw,thick,minimum size=0.45cm,inner sep=0pt,font=\\scriptsize},\n"
                "  hid/.style={circle,draw,thick,dashed,fill=gray!25,minimum size=0.45cm,inner sep=0pt,font=\\scriptsize},\n"
                "  arr/.style={-{Stealth[length=1.6mm]},thick},\n"
                "  arr2/.style={{Stealth[length=1.6mm]}-{Stealth[length=1.6mm]},thick,red!70!black}}\n")

    def to_tikz(self, xscale=0.8, yscale=0.8, with_preamble=True):
        """
        TikZ code of one row (DAG | ancestral ADMG | explicit W' | implicit
        Omega), ready to be put in a figure environment.
        """
        xs = [p[0] for p in self.pos.values()]
        shift = (max(xs) - min(xs) + 1.0) * xscale + 0.4
        top = max(p[1] for p in self.pos.values()) + 0.8
        no = len(self.observed)
        pre = self.tikz_preamble() if with_preamble else ""
        return (
            pre
            + f"\\begin{{tikzpicture}}[x={xscale}cm,y={yscale}cm,baseline=(current bounding box.center)]\n"
            f"\\node[font=\\scriptsize] at ({np.mean(xs):.2f},{top:.2f}) {{DAG}};\n"
            + self._tikz_graph(True, xscale, yscale) + "\n"
            f"\\begin{{scope}}[xshift={shift:.2f}cm]\n"
            f"\\node[font=\\scriptsize] at ({np.mean(xs):.2f},{top:.2f}) {{ancestral ADMG}};\n"
            + self._tikz_graph(False, xscale, yscale) + "\n\\end{scope}\n"
            "\\end{tikzpicture}\\hfill\n"
            "\\begin{tikzpicture}[baseline=(current bounding box.center)]\n"
            + self._tikz_matrix(self.explicit_matrix(), self.order, no, "explicit $W'$") + "\n"
            "\\end{tikzpicture}\\hfill\n"
            "\\begin{tikzpicture}[baseline=(current bounding box.center)]\n"
            + self._tikz_matrix(self.omega_pattern(), self.observed, no, "implicit $\\Omega$") + "\n"
            "\\end{tikzpicture}\n"
        )
