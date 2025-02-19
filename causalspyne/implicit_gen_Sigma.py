"""
"""
import numpy as np
from causalspyne.implicit_omega_condition_on_w import gen_joint_w_omega, get_max_degree
from causalspyne.implicit_omega import get_extreme_eigenvalue


def cov_to_corr(covariance_matrix):
    # Calculate the standard deviations
    std_devs = np.sqrt(np.diag(covariance_matrix))

    # Create a diagonal matrix of inverse standard deviations
    inv_std_devs = np.diag(1 / std_devs)

    # Calculate the correlation matrix
    correlation_matrix = inv_std_devs @ covariance_matrix @ inv_std_devs

    return correlation_matrix


def congruent(mat_op, mat):
    """
    congruent transform is defined to be (mat_op)^T mat (mat_op)
    """
    first2 = np.matmul(mat_op.T, mat)
    mat_rst = np.matmul(first2, mat_op)
    return mat_rst


def gen_sigma_y(max_w=0.5):
    """
    generate Sigma matrix according to $${(I-W)}^{-1}\Omega{(I-W)}^{-1}$$
    """
    mat_w_binary, mat_omega = gen_joint_w_omega()
    mat_weight = np.random.uniform(
        low=-max_w, high=max_w, size=mat_w_binary.shape)

    mat_w = mat_w_binary * mat_weight
    print(f"adjacency weighted: {mat_w}")

    dmax = get_max_degree(mat_w)
    kernel = np.identity(mat_w.shape[0]) - mat_w
    inv_id_minus_w = np.linalg.inv(kernel)
    mat_sigma = congruent(inv_id_minus_w.T, mat_omega)
    max_omega = np.max(mat_omega)
    return mat_sigma, dmax, max_omega


def gen_spectrum():
    mat_sigma, dmax, max_omega = gen_sigma_y()
    print(f"sigma: {mat_sigma}")
    inv_sigma = np.linalg.inv(mat_sigma)
    eigv_max, _ = get_extreme_eigenvalue(inv_sigma)
    print(f"max eigen value: {eigv_max}")
    print(f"max degree: {dmax}, max_omega: {max_omega}")
    mat_precision = cov_to_corr(inv_sigma)
    eigv_max, _ = get_extreme_eigenvalue(mat_precision)
    print(f"max eigen value: {eigv_max}")
    return eigv_max


if __name__ == "__main__":
    gen_spectrum()
