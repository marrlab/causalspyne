def is_dag(adjacency_matrix):
    n = len(adjacency_matrix)
    visited = set()
    path = set()

    def dfs(node):
        visited.add(node)
        path.add(node)

        for neighbor in range(n):
            if adjacency_matrix[node][neighbor] == 1:
                if neighbor in path:
                    return False
                if neighbor not in visited:
                    if not dfs(neighbor):
                        return False

        path.remove(node)
        return True

    for node in range(n):
        if node not in visited:
            if not dfs(node):
                return False

    return True
