import numpy as np

def estimate_spectrum_radius(arr_df):
    mat_corr = np.corrcoef(arr_df.T)
    mat_cov = np.cov(arr_df.T)
    mat_precision = np.linalg.inv(mat_cov)
    arr_eig_values, eigvec = np.linalg.eig(mat_precision)
    return np.max(arr_eig_values)
