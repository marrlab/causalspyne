# ProblemSetApproximateCausalDiscovery

```
from causalspyne import gen_partially_observed


gen_partially_observed(size_micro_node_dag=4,     
                       num_macro_nodes=4,
                       degree=2,  # average vertex/node degree
                       list_confounder2hide=[0.5, 0.9], # choie of confounder to hide: percentile or index of all toplogically sorted confounders 
                       num_sample=200)
```
