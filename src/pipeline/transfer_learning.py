
import numpy as np
from scipy.linalg import inv, sqrtm

def get_align_mat(X):
    """Take a data matrix of shape (trials, channels, samples) and return the alignment matrix
    of shape (channels, channels)."""
    covs = np.empty((X.shape[0], X.shape[1], X.shape[1]))
    for trial in range(X.shape[0]):
        covs[trial, :, :] = np.cov(X[trial, :, :])
    ref_mat = np.mean(covs, axis=0)
    return inv(sqrtm(ref_mat))
    
    
def align(X, align_mat):
    """Take a data matrix of shape (trials, channels, samples) and an alignment matrix
    of shape (channels, channels), align each trial and return the resulting matrix."""
    
    X_aligned = np.empty(X.shape)
    for trial in range(X.shape[0]):
        X_aligned[trial, :, :] = np.matmul(align_mat, X[trial, :, :])
    return X_aligned
