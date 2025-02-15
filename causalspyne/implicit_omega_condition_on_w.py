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


def gen_joint_w_omega():
    mat_w = Erdos_Renyi_PLP(default_rng(0))(num_nodes=6, degree=2)
    max_omega = 0.7
    mat, rho = gen_omega(3, max_omega)
    print(mat)
    max_eig, min_eig = get_extreme_eigenvalue(mat)
    print(f"min, max eigenvalue {min_eig, max_eig}")
    print(rho, 2 * rho)


if __name__ == "__main__":
    gen_joint_w_omega()
