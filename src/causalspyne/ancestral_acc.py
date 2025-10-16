"""
ancestral accuracy calculate the percentage of correct prediction
among all pairwise variables' ancestral relationships (binary)
"""
from itertools import combinations
import warnings

from causalspyne.dag2ancestral import DAG2Ancestral


def ancestral_acc(true_dag, pred_order, list_hidden_nodes=None):
    """
    Parameters:
    true_dag: Target causalspyne graph
    pred_order:
    ISSUE #34, the pairwise relationship can be + or - ancestral, but can also be no ancestral relationship
    list_hidden_nodes (list): list of nodes in DAG to hide, here we use pairwise combinations of pred_order, which
    already assumes the algorithm has a clear + order only. 
    Returns:
    float:
    """
    dag2ancestral = DAG2Ancestral(true_dag.mat_adjacency)
    dag2ancestral.pre_cal_n_hop()
    size_dag = true_dag.num_nodes
    if list_hidden_nodes is not None:
        n_vars = size_dag - len(list_hidden_nodes)
    else:
        n_vars = size_dag
    if n_vars != len(pred_order):
        warnings.warn(f"predicted causal order {pred_order} does not \
                      have the same number of observables!, hidden are: {list_hidden_nodes} \
                      now forcing number of variables to be the lengh of pred_order")
        n_vars = len(pred_order)

    pairwise_combinations = list(combinations(pred_order, 2))
 
    n_correct = 0
    for pair in pairwise_combinations:
        ancestor, offspring = pair
        if dag2ancestral.is_ancestor(ancestor, offspring):
            n_correct += 1
    num_combos = n_vars * (n_vars - 1) /2
    return float(n_correct) / num_combos
