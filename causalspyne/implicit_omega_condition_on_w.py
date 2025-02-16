"""
This module is used to generate the implicit omega matrix.

To generate a symmetric matrix,
    - A+A^T
    - np.tril(A) + np.tril(A, -1).T
    np.tril(A, -1) returns a copy of the input array A with elements above the
    first sub-diagonal zeroed out
"""
import numpy as np
from numpy.random import default_rng
from causalspyne.erdo_renyi_plp import Erdos_Renyi_PLP
from causalspyne.implicit_omega import gen_omega, get_extreme_eigenvalue


def gen_joint_w_omega(p_ob_v=4, ave_degree=2, max_omega=0.7):
    mat_w = Erdos_Renyi_PLP(default_rng(0))(num_nodes=p_ob_v,
                                            degree=ave_degree)
    print(f"adjacency: \n {mat_w}")
    mat_omega, rho = gen_omega(p_ob_v, max_omega)
    print(f"omega: \n {mat_omega}")
    max_eig, min_eig = get_extreme_eigenvalue(mat_omega)
    print(f"min, max eigenvalue {min_eig, max_eig}")
    print(f"rho, 2*rho: {rho}, {2 * rho}")
    return mat_w, mat_omega


if __name__ == "__main__":
    gen_joint_w_omega()
