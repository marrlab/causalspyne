"""
This module is used to generate the implicit omega matrix.

To generate a symmetric matrix,
    - A+A^T
    - np.tril(A) + np.tril(A, -1).T
    np.tril(A, -1) returns a copy of the input array A with elements above the
    first sub-diagonal zeroed out
"""
import numpy as np


def is_diagonal(matrix):
    return np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0


def is_diag_dom(mat):
    for i_row, row in enumerate(mat):
        energy_diag = np.abs(mat[i_row, i_row])
        energy_row_off_diag = np.sum(np.abs(row)) - energy_diag
        flag = energy_diag <= energy_row_off_diag
        if flag:
            return False
    return True


def gen_omega(n_ob_v, max_omega, delta=1.0, p_sparse=0.1):
    """
    :n_ob_v: number of observed variables
    :max_omega: maximum value of omega
    :returns: None
    """
    assert max_omega > 0
    # Generate Bernoulli random numbers
    mat_bernoulli = np.random.binomial(
        n=1, p=p_sparse, size=(n_ob_v, n_ob_v))
    mat_omega = np.random.uniform(
        low=0.1, high=max_omega, size=(n_ob_v, n_ob_v))
    mat_sparse_omega = mat_bernoulli * mat_omega
    mat_sym = 0.5 * (mat_sparse_omega + mat_sparse_omega.T)

    if is_diagonal(mat_sym):
        rng = np.random.default_rng()
        sampled_row_ind = rng.integers(low=0, high=n_ob_v - 1, size=1)
        off_diag_v = mat_omega[sampled_row_ind, sampled_row_ind + 1]
        mat_sym[sampled_row_ind, sampled_row_ind + 1] = off_diag_v
        mat_sym[sampled_row_ind + 1, sampled_row_ind] = off_diag_v
    for i_row, row in enumerate(mat_sym):
        old_diag_v = np.abs(mat_sym[i_row, i_row])
        energy_row_off_diag = np.sum(np.abs(row)) - np.abs(old_diag_v)
        mat_sym[i_row, i_row] = delta + energy_row_off_diag
    assert is_diag_dom(mat_sym)
    return mat_sym, delta + max_omega


def get_extreme_eigenvalue(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.max(np.real(eigenvalues)), np.min(np.real(eigenvalues))


if __name__ == "__main__":
    max_omega = 0.7
    mat, rho = gen_omega(3, max_omega)
    print(mat)
    max_eig, min_eig = get_extreme_eigenvalue(mat)
    print(f"min, max eigenvalue {min_eig, max_eig}")
    print(rho, 2 * rho)
