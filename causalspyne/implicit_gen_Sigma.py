"""
"""
import numpy as np
from causalspyne.implicit_omega_condition_on_w import gen_joint_w_omega


def gen_Sigma():
    mat_w, mat_omega = gen_joint_w_omega()
    left = np.linalg.inv(np.identiy(mat_w.ndim)-mat_w)
    temp = np.matmul(left, mat_omega)
    sigma = np.matmul(temp, np.identiy(mat_w.ndim)-mat_w))


if __name__ == "__main__":
    gen_Sigma()
