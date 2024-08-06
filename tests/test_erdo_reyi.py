from causalspyne.dag_gen import Erdos_Renyi_PLP
from causalspyne.is_dag import is_dag


def test_erdos_renyi():
    mat1 = Erdos_Renyi_PLP()(num_nodes=3, degree=2)
    is_dag(mat1)
