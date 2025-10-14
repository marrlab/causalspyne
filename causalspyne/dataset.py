import numpy as np
from numpy.random import default_rng

from causalspyne.dag_interface import MatDAG
from causalspyne.noise_idiosyncratic import Bernoulli
from causalspyne.data_gen import  DataGen

def simpson():
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

    data_gen = DataGen(dag, edge_model=None,
                   idiosynchratic={0:Bernoulli(rng=default_rng(), params={"p":0.2})})
    print(data_gen.gen(200))

simpson()
