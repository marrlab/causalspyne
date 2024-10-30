from numpy.random import default_rng

from causalspyne.erdo_renyi_plp import Erdos_Renyi_PLP
from causalspyne.is_dag import is_dag


def test_erdos_renyi():
    mat1 = Erdos_Renyi_PLP(default_rng(0))(num_nodes=3, degree=2)
    is_dag(mat1)
