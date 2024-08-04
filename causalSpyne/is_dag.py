"""
DFS to check if is DAG, use python inner function to access global variable
"""


def is_dag(adjacency_matrix):
    """
    adjaceny_matrx[u][v] \\neq 0 \\iff u <- v
    """
    n = len(adjacency_matrix)
    set_visited = set()
    set_path = set()  # A->[B]->[C]->A

    def dfs(ind_node):
        set_visited.add(ind_node)
        set_path.add(ind_node)

        for arrow_head in range(n):
            if adjacency_matrix[arrow_head][ind_node] != 0:  # arrow head
                if arrow_head in set_path:  # # A->B->C->A
                    return False  # not a DAG, bottom level return
                if arrow_head not in set_visited:
                    if not dfs(arrow_head):  # recursion
                        return False  # recursive return
        # exit of recursion: if arrow_head is visited
        set_path.remove(ind_node)
        return True

    for ind_node in range(n):
        if ind_node not in set_visited:
            if not dfs(ind_node):
                return False

    return True
