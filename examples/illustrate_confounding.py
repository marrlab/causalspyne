"""
Illustrate hidden confounding: DAG, ancestral ADMG, explicit block matrix W'
and the implied Omega pattern, as a Matplotlib figure and as TikZ code.

    uv run python examples/illustrate_confounding.py --hide 1
    uv run python examples/illustrate_confounding.py --hide 2 5 --out fig_hidden_intermediate
    uv run python examples/illustrate_confounding.py --random 7 --seed 0 --hide 0

With no --random the 7-vertex DAG of the CLeaR 2027 paper is used. Output files:
<out>.pdf, <out>.png and <out>.tex (the TikZ row, with the preamble).
"""
import argparse

import numpy as np

from causalspyne.illustration import ConfoundingIllustration

# edges parent -> child of the example DAG of the paper (Figure 1)
PAPER_EDGES = [(1, 0), (1, 2), (1, 5), (2, 0), (2, 4), (5, 3), (5, 4), (6, 3),
               (6, 4), (0, 4), (3, 4)]


# hand-set layout of Figure 1 of the paper (default layout is ConfoundingIllustration's layered one)
PAPER_POS = {1: (1.5, 3), 2: (1.0, 2), 5: (2.2, 2), 6: (3.4, 2), 0: (-0.6, 1),
             3: (2.2, 1), 4: (1.5, 0)}


def paper_dag():
    adj = np.zeros((7, 7), dtype=int)
    for parent, child in PAPER_EDGES:
        adj[child, parent] = 1
    return adj


def random_dag(num_nodes, seed, prob=0.3):
    rng = np.random.default_rng(seed)
    low = np.tril(rng.random((num_nodes, num_nodes)) < prob, -1)
    perm = rng.permutation(num_nodes)
    return low[np.ix_(perm, perm)].astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hide", type=int, nargs="+", default=[1],
                        help="global indices of the hidden vertices")
    parser.add_argument("--random", type=int, default=None, metavar="N",
                        help="use a random DAG with N vertices instead of the paper example")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paper-layout", action="store_true",
                        help="use the hand-set vertex positions of the paper figure (paper DAG only)")
    parser.add_argument("--out", default="confounding_illustration")
    args = parser.parse_args()
    adj = random_dag(args.random, args.seed) if args.random else paper_dag()
    pos = PAPER_POS if (args.paper_layout and not args.random) else None
    ill = ConfoundingIllustration(adj, hidden=args.hide, pos=pos)
    fig = ill.to_matplotlib()
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    with open(args.out + ".tex", "w") as handle:
        handle.write(ill.to_tikz())
    print("bidirected edges:", ill.bidirected)
    print("directed edges among observed:", sorted(ill.directed))
    print("wrote", args.out + ".pdf, .png, .tex")


if __name__ == "__main__":
    main()
