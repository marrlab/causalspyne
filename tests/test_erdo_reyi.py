from causalSpyne.dag_gen import Erdos_Renyi


def test_erdos_renyi():
    Erdos_Renyi()(num_nodes=3, degree=2, list_weight_range=[2, 3])
