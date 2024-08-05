import numpy as np
from scipy import stats

def gen_weight_matrix(n=4):
    df = n + 1 
    scale_matrix = np.eye(n)
    weight = stats.wishart.rvs(df=df, scale=scale_matrix, size=1)
    return weight