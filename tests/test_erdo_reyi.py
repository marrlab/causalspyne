from causalSpyne.dag_gen import Erdos_Renyi
from causalSpyne.is_dag import is_dag


def test_erdos_renyi():
    mat1, mat2 = Erdos_Renyi()(num_nodes=3, degree=2, list_weight_range=[2, 3])
    is_dag(mat1)
    is_dag(mat2)
