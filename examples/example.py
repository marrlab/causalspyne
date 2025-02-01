from causalspyne import gen_partially_observed
from causalspyne.utils_cov_precision import estimate_spectrum_radius


subview = gen_partially_observed(size_micro_node_dag=3,
                                 num_macro_nodes=3,
                                 degree=2,
                                 list_confounder2hide=[0],
                                 num_sample=200,
                                 graphviz=True)

radius = estimate_spectrum_radius(subview.data)
print(radius)
