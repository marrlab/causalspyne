"""
This module is used to generate the implicit omega matrix.

To generate a symmetric matrix,
    - A+A^T
    - np.tril(A) + np.tril(A, -1).T
    np.tril(A, -1) returns a copy of the input array A with elements above the
    first sub-diagonal zeroed out
"""
import numpy as np
from causalspyne.implicit_omega_condition_on_w import gen_joint_w_omega


def gen_Sigma():
    mat_w, mat_omega = gen_joint_w_omega()


if __name__ == "__main__":
    gen_Sigma()
