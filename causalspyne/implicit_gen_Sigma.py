"""
"""
import numpy as np
from causalspyne.implicit_omega_condition_on_w import gen_joint_w_omega


def gen_Sigma():
    mat_w, mat_omega = gen_joint_w_omega()
    np.linalg.inv(np.identiy(mat_w.ndim)-mat_w)


if __name__ == "__main__":
    gen_Sigma()
