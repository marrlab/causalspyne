"""
test if ancestral acc runs
"""

import numpy as np
from causalspyne.ancestral_acc import ancestral_acc
from causalspyne import gen_partially_observed


def test_ancestral_acc():
    """
    test if ancestral accuracy runs
    """
    subview = gen_partially_observed(
        size_micro_node_dag=3,
        num_macro_nodes=2,
        degree=2,  # average vertex/node degree
        list_confounder2hide=[
            0.5,
            1.0,
        ],  # choie of confounder to hide: percentile or index of all toplogically sorted confounders
        num_sample=200,
        rng=np.random.default_rng(1),
        graphviz=False
    )
    pred_order_inds = \
        [subview.dag._dict_node_names2ind[name]
         for name in subview.dag.list_node_names]
    acc = ancestral_acc(true_dag=subview.dag,
                        pred_order=pred_order_inds)

    print(f"ancestral acc: {acc}")
