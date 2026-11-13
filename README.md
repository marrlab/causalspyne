# CausalSpyne
[![PyPI version](https://badge.fury.io/py/causalspyne.svg)](https://badge.fury.io/py/causalspyne)  [![test coverage](https://marrlab.github.io/causalspyne/coverage-badge.svg)](https://marrlab.github.io/causalspyne/)

A Python package for simulating data from confounded causal models.


## Quick start
Install with: `pip install causalspyne`

Generate some data:
```
from causalspyne import gen_partially_observed


gen_partially_observed(size_micro_node_dag=4,
                       num_macro_nodes=4,
                       degree=2,  # average vertex/node degree
                       list_confounder2hide=[0.5, 0.9], # choie of confounder to hide: percentile or index of all toplogically sorted confounders
                       num_sample=200,
                       output_dir="output",
                       rng=0)
```

## Submodules

```
git submodule update --init
```

## Illustrating hidden confounding (DAG, ancestral ADMG, block matrices)

`causalspyne.illustration.ConfoundingIllustration(adj, hidden)` draws, for a DAG and a set of hidden vertices, the
DAG, the ancestral ADMG over the observed vertices (bidirected edges in red), the block weight matrix
`W' = [[W_O, W_OU], [W_UO, W_UU]]` of the explicit parameterization (observed vertices first; row = child,
column = parent) and the nonzero pattern of the implied noise covariance `Omega`. `to_matplotlib()` returns a
Matplotlib figure, `to_tikz()` returns TikZ code for a LaTeX paper. See `examples/illustrate_confounding.py`.
