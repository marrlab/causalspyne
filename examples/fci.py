from causallearn.search.ConstraintBased.FCI import fci
from causalspyne import gen_partially_observed


arr_data, node_names = gen_partially_observed(size_micro_node_dag=3,
                       num_macro_nodes=2,
                       degree=2,  # average vertex/node degree
                       list_confounder2hide=[0.5, 1.0], # choie of confounder to hide: percentile or index of all toplogically sorted confounders
                       num_sample=200)

# default parameters
g, edges = fci(arr_data)


print(g.graph) # numpy array
# visualization
from causallearn.utils.GraphUtils import GraphUtils

pdy = GraphUtils.to_pydot(g, labels=node_names)
pdy.write_png('output_causallearn.png')
