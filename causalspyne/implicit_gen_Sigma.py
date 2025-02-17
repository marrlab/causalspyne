"""
"""
import numpy as np
from causalspyne.implicit_omega_condition_on_w import gen_joint_w_omega, get_max_degree
from causalspyne.implicit_omega import get_extreme_eigenvalue


def congruent(mat_op, mat):
    """
    congruent transform is defined to be (mat_op)^T mat (mat_op)
    """
    first2 = np.matmul(mat_op.T, mat)
    mat_rst = np.matmul(first2, mat_op)
    return mat_rst


def gen_sigma_y():
    """
    generate Sigma matrix according to $${(I-W)}^{-1}\Omega{(I-W)}^{-1}$$
    """
    mat_w, mat_omega = gen_joint_w_omega()
    dmax = get_max_degree(mat_w)
    kernel = np.identity(mat_w.shape[0]) - mat_w
    inv_id_minus_w = np.linalg.inv(kernel)
    mat_sigma = congruent(inv_id_minus_w.T, mat_omega)
    return mat_sigma, dmax


if __name__ == "__main__":
    mat_sigma, dmax = gen_sigma_y()
    print(f"sigma: {mat_sigma}")
    inv_sigma = np.linalg.inv(mat_sigma)
    eigv_max, _ = get_extreme_eigenvalue(inv_sigma)
    print(f"max eigen value: {eigv_max}")
    print(f"max degree: {dmax}")
