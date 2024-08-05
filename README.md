# ProblemSetApproximateCausalDiscovery

ˋˋˋ

from causalSpyne import gen_partially_observed

gen_partially_observed(degree=2,  # average degree

                       list_confounder2hide=[0.5, 0.9], # percentile of confounder in toplogical order to hide
                       
                       size_micro_node_dag=4,
                       
                       num_macro_nodes=4,
                       
                       num_sample=200)
ˋˋˋ

