import numpy as np
from numpy.random import default_rng

from causalspyne.dag_interface import MatDAG
from causalspyne.noise_idiosyncratic import Idiosyncratic
from causalspyne.data_gen import  DataGen

def simpson(size_sample=200):
    # 0 as confounder: 0->1, 0->2, 1->2
    mat_weighted_adjacency = np.array(
        [
            # 0 1 2 3
            [0, 0, 0],  # 0: confounder, root variable
            [1, 0, 0],  # 1: 0->1
            [1, 1, 0],  # 2: 0->2, 1->2
        ]
    )

    dag = MatDAG(mat_weighted_adjacency,
                name_prefix="V",
                rng=default_rng())

    confounder = Idiosyncratic(rng=default_rng(),
                               class_name="Bernoulli",
                               dict_params={"p":0.2})

    data_gen = DataGen(dag, edge_model=None,
                       idiosynchratic={0:confounder})

    print(data_gen.gen(size_sample))
