import numpy as np
from causalSpyne.utils_topological_sort import topological_sort

def test_topological_sort():
    """
    adj must be lower triangular
    """
    adj_matrix = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    adj_matrix = adj_matrix.transpose()

    try:
        order = topological_sort(adj_matrix)
        print("Topological order:", order)
    except ValueError as e:
        print(e)
