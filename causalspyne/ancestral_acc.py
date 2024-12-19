"""
ancestral accuracy calculate the percentage of correct prediction
among all pairwise variables' ancestral relationships (binary)
"""
from itertools import combinations

from causalspyne.dag2ancestral import DAG2Ancestral


def ancestral_acc(true_dag, pred_order, list_hidden_nodes=None):
    """

    Parameters:
    true_dag: Target causalspyne graph
    list_hidden_nodes (list): list of nodes in DAG to hide
    Returns:
    int:
    """
    dag2ancestral = DAG2Ancestral(true_dag.mat_adjacency)
    if true_hidden_nodes is not None:
        n_obs = len(true_dag) - len(true_hidden_nodes)
    else:
        n_obs =
    if n_obs != len(pred_order[1]):
        raise ValueError("predicted causal order does not \
                         have the same number of observables!")

    pairwise_combinations = list(combinations(pred_order, 2))

    n_correct = 0
    for pair in pairwise_combinations:
        ancestor, offspring = pair
        if dag2ancestral.is_ancestor(ancestor, offspring):
            n_correct += 1
    return float(n_correct) / n_obs
