import numpy as np

from causalspyne.draw_dags import draw_dags_nx


def test_draw_dag_nx():
    # Create an example adjacency matrix for a DAG
    adj_matrix = np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
    draw_dags_nx(adj_matrix=adj_matrix, show=False)
