import numpy as np

def load_features(fname):
    features = np.load(fname)
    return features

def compute_correlation(mat1, mat2, corr_func):
    assert mat1.shape == mat2.shape
    n_entries = mat1.shape[0]
    
    mat1_upr = mat1[np.triu_indices(n_entries, 1)]
    mat2_upr = mat2[np.triu_indices(n_entries, 1)]
    
    r, p = corr_func(mat1_upr, mat2_upr)
    return r

