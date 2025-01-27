import numpy as np
from causalspyne import gen_partially_observed, ordered_ind_col2global_ind
from causalspyne.main import ordered_ind_col2global_ind
from causalspyne.ancestral_acc import ancestral_acc

import graphviz
import lingam
from lingam.utils import print_causal_directions, print_dagc, make_dot


subview, node_names, dag, subview_global_inds  = gen_partially_observed(
    size_micro_node_dag=3,
    num_macro_nodes=2,
    degree=2,  # average vertex/node degree
    list_confounder2hide=[
        0.5,
        1.0,
    ],  # choie of confounder to hide: percentile or index of all toplogically sorted confounders
    num_sample=200,
    rng=np.random.default_rng(1),
    graphviz=True
)


model = lingam.BottomUpParceLiNGAM()
model.fit(subview.data)

print(f"predicted observable order:{model.causal_order_}")

nested_list = model.causal_order_

print(f"causal order {nested_list}")

flat_list = [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]


pred_obs_order = ordered_ind_col2global_ind(inds_cols=flat_list,
                                            subview_global_inds=subview_global_inds)


acc = ancestral_acc(dag, pred_order=pred_obs_order)

print(f"ancestral acc:{acc}")


make_dot(model.adjacency_matrix_)
