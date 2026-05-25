import numpy as np


def _can_reach(mat, src, dst):
    """Return True if there is a directed path from src to dst in mat."""
    n = mat.shape[0]
    visited = set()
    stack = [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node in visited:
            continue
        visited.add(node)
        # children of node: columns where mat[:, node] != 0
        for child in np.where(mat[:, node] != 0)[0]:
            if child not in visited:
                stack.append(child)
    return False


class RandTopoOrderDAG:
    """
    Generate a random DAG by sampling a random topological order, then
    independently including each forward edge with probability
    degree / (num_nodes - 1).

    Guaranteed to produce a valid DAG: edges only go from earlier to
    later positions in the sampled order, so no cycle is possible.

    Matrix convention matches Erdos_Renyi_PLP: entry (i, j) means j -> i.
    """

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, num_nodes, degree):
        prob = float(degree) / (num_nodes - 1)
        topo_order = self.rng.permutation(num_nodes)  # topo_order[k] = node at position k
        mat = np.zeros((num_nodes, num_nodes))
        for k in range(num_nodes):
            for l in range(k + 1, num_nodes):
                if self.rng.random() < prob:
                    src, dst = topo_order[k], topo_order[l]
                    mat[dst, src] = 1.0  # src -> dst
        return mat


class RootConfounderDAG(RandTopoOrderDAG):
    """
    Extends RandTopoOrderDAG so that every root node (no incoming edges)
    is guaranteed to be a confounder (at least 2 outgoing edges).

    After base generation, any root with fewer than 2 children gets extra
    edges added to nodes that cannot reach the root (cycle-safe by construction).
    """

    def __call__(self, num_nodes, degree):
        mat = super().__call__(num_nodes, degree)
        root_nodes = list(np.where(mat.sum(axis=1) == 0)[0])
        for root in root_nodes:
            current_children = list(np.where(mat[:, root] > 0)[0])
            needed = max(0, 2 - len(current_children))
            if needed == 0:
                continue
            # exclude nodes that can reach root (adding root->n would create a cycle)
            candidates = [n for n in range(num_nodes)
                          if n != root
                          and n not in current_children
                          and not _can_reach(mat, n, root)]
            if len(candidates) < needed:
                continue
            targets = self.rng.choice(candidates, size=needed, replace=False)
            for t in targets:
                mat[t, root] = 1.0
        return mat
