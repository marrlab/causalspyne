import numpy as np
from causalspyne.utils_causallearn_g2ancestral import get_causalearn_order
from causalspyne import gen_partially_observed, ordered_ind_col2global_ind
from causalspyne.ancestral_acc import ancestral_acc

from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.graph.NodeType import NodeType

# visualization
from causallearn.utils.GraphUtils import GraphUtils
# Visualization using pydot
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io


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
    graphviz=True
)
print(f"subview global inds: {subview.col_inds}")
G, K = GIN(subview.data)

pred_obs_order, pred_latent_order = get_causalearn_order(G, subview.node_names)
print(f"predicted observable order:{pred_obs_order}")

ind_cols = [int(name.removeprefix("X")) for name in pred_obs_order]
# pred_obs_order_aligned = ordered_ind_col2global_ind(ind_cols, subview_global_inds)
acc = ancestral_acc(subview.dag, pred_order=ind_cols)

print(f"acc: {acc}")

print(f"latent cluster order: {K}, type: {type(K)}, len: {len(K)}")

print(G.graph)  # numpy array of PAG,
# see https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/FCI.html#usage

# FIXME: directly providing pred_obs_order to "to_pydot" cause inconsistency of GraphUtils
pyd = GraphUtils.to_pydot(G)
pyd.write_png("output_causallearn_gin.png")

# for showing image
# tmp_png = pyd.create_png(f="png")
# fp = io.BytesIO(tmp_png)
# img = mpimg.imread(fp, format='png')
# plt.axis('off')
# plt.imshow(img)
# plt.show()
