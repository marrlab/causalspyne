from numpy.random import default_rng

from causalspyne import gen_partially_observed


def test_main(tmp_path):
    gen_partially_observed(
        degree=2,
        list_confounder2hide=[0.5, 0.9],
        size_micro_node_dag=4,
        num_macro_nodes=4,
        num_sample=200,
        output_dir=tmp_path,
        rng=default_rng(0),
    )


def test_main_accepts_integer_seed_and_output_path_without_trailing_slash(tmp_path):
    output_dir = tmp_path / "output"
    gen_partially_observed(
        degree=2,
        list_confounder2hide=[0.5, 0.9],
        size_micro_node_dag=4,
        num_macro_nodes=4,
        num_sample=200,
        output_dir=output_dir,
        rng=0,
        plot=False,
    )

    assert list(output_dir.glob("ground_truth_dag_*d.csv"))
    assert not list(output_dir.glob("graph_compare_*"))
