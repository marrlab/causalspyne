from causallearn.search.ConstraintBased.FCI import fci
from causalspyne import gen_partially_observed


data = gen_partially_observed(size_micro_node_dag=4,
                       num_macro_nodes=4,
                       degree=2,  # average vertex/node degree
                       list_confounder2hide=[0.5, 1.0], # choie of confounder to hide: percentile or index of all toplogically sorted confounders
                       num_sample=200)

# default parameters
g, edges = fci(data)

# visualization
from causallearn.utils.GraphUtils import GraphUtils

pdy = GraphUtils.to_pydot(g)
pdy.write_png('simple_test.png')
