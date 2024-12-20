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
    list_hidden_nodes (list): list of nodes in DAG to hide
    Returns:
    int:
    """
    dag2ancestral = DAG2Ancestral(true_dag.mat_adjacency)
    dag2ancestral.pre_cal_n_hop()
    size_dag = true_dag.num_nodes
    if list_hidden_nodes is not None:
        n_obs = size_dag - len(list_hidden_nodes)
    else:
        n_obs = size_dag
    if n_obs != len(pred_order):
        warnings.warn(f"predicted causal order {pred_order} does not \
                      have the same number of observables!, hidden are: {list_hidden_nodes}")

    pairwise_combinations = list(combinations(pred_order, 2))

    print(f"available keys and vals are: {true_dag._dict_node_names2ind}")
    print(f"generated keys (pred order) are: {pred_order}")
    
    n_correct = 0
    for pair in pairwise_combinations:
        ancestor, offspring = pair
        id_ancestor = true_dag._dict_node_names2ind[ancestor]
        id_offspring = true_dag._dict_node_names2ind[offspring]
        if dag2ancestral.is_ancestor(id_ancestor, id_offspring):
            n_correct += 1
    return float(n_correct) / n_obs
