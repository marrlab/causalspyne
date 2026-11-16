import os, sys
import numpy as np
from causalspyne.explicit_confounding_gen import random_instance, pair_loadings, omega_from_loadings, add_bows, sample_uniform_explicit


def test_instance_is_bowfree_dag_and_bows_added_acyclically():
    rng = np.random.default_rng(0)
    W, pairs = random_instance(rng, 10, 0.1, 0.5)
    assert all(W[i, j] == 0 and W[j, i] == 0 for i, j in pairs)
    assert all(sum(v in p for p in pairs) <= 2 for v in range(10))
    W2 = add_bows(rng, W, pairs, 0.1, 0.5)
    assert (np.abs(np.linalg.eigvals(W2)) < 1e-9).all()                    # nilpotent: still a DAG
    assert sum(W2[j, i] != 0 or W2[i, j] != 0 for i, j in pairs) >= max(1, len(pairs) // 2)


def test_equal_loadings_give_diagonally_dominant_omega():
    pairs = [(0, 1), (1, 2), (0, 3)]
    theta = np.array([0.5, -0.3, 0.2])
    Om = omega_from_loadings(pair_loadings(theta, pairs, 4))
    assert np.isclose(Om[0, 1], 0.5) and np.isclose(Om[1, 2], -0.3) and np.isclose(Om[0, 3], 0.2)
    assert np.isclose(Om[0, 0], 1 + 0.5 + 0.2) and np.isclose(Om[1, 1], 1 + 0.5 + 0.3)
    assert (np.diag(Om) - np.abs(Om - np.diag(np.diag(Om))).sum(1) >= 1 - 1e-12).all()


def test_sample_covariance_matches_implicit_formula():
    rng = np.random.default_rng(1)
    W, pairs = random_instance(rng, 5, 0.1, 0.5)
    Lam = pair_loadings(rng.uniform(-0.7, 0.7, len(pairs)), pairs, 5)
    X = sample_uniform_explicit(W, Lam, 200000, rng, standardize=False)
    A = np.linalg.inv(np.eye(5) - W)
    Sigma = A @ omega_from_loadings(Lam) @ A.T
    assert np.abs(np.cov(X.T) - Sigma).max() < 0.05 * np.abs(Sigma).max()
