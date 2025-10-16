import numpy as np
from scipy import stats


def gen_weight_matrix(rng, n=4):
    df = n + 1
    scale_matrix = np.eye(n)
    weight = stats.wishart(df=df, scale=scale_matrix, seed=rng).rvs(size=1)
    if weight.shape == ():
        weight = weight[None, None]
    return weight
