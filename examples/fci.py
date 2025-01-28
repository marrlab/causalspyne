import numpy as np
from causallearn.search.ConstraintBased.FCI import fci
from causalspyne import gen_partially_observed

# visualization
from causallearn.utils.GraphUtils import GraphUtils


subview, dag, subview_global_inds  = gen_partially_observed(
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

# default parameters
g, edges = fci(subview.data)


print(g.graph)  # numpy array

"""
see (https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/FCI.html#returns) for how to interpret FCI output

g.graph with rng(0)
array([[0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 2],
      [0, 0, 2, 0]])
here node4-5 connection type 2 edge means non-determined o-o type edge
g.graph with rng(1)
[[0 2 0 0 2]
[2 0 2 0 2]
[0 2 0 2 2]
[0 0 2 0 0]
[2 2 2 0 0]]

g.graph with rng(2)
1 means directed edge
[[0 2 0 0 0 0]
[1 0 1 0 0 0]
[0 2 0 2 2 0]
[0 0 2 0 0 0]
[0 0 1 0 0 1]
[0 0 0 0 2 0]]
"""
pdy = GraphUtils.to_pydot(g, labels=subview.node_names)
pdy.write_png("output_causallearn_fci.png")
