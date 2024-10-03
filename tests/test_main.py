from numpy.random import default_rng

from causalspyne import gen_partially_observed


def test_main():
    gen_partially_observed(
        degree=2,
        list_confounder2hide=[0.5, 0.9],
        size_micro_node_dag=4,
        num_macro_nodes=4,
        num_sample=200,
        rng=default_rng(0),
    )
