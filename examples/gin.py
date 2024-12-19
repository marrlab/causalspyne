import numpy as np
from causalspyne import gen_partially_observed
from causallearn.search.HiddenCausal.GIN.GIN import GIN

# visualization
from causallearn.utils.GraphUtils import GraphUtils
# Visualization using pydot
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io


arr_data, node_names = gen_partially_observed(
    size_micro_node_dag=3,
    num_macro_nodes=2,
    degree=2,  # average vertex/node degree
    list_confounder2hide=[
        0.5,
        1.0,
    ],  # choie of confounder to hide: percentile or index of all toplogically sorted confounders
    num_sample=200,
    rng=np.random.default_rng(2),
    graphviz=True
)

G, K = GIN(arr_data)

pyd = GraphUtils.to_pydot(G)
pyd.write_png("output_causallearn_gin.png")

# for showing image
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
# plt.show()