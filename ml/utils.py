import numpy as np


def log_clip(x, clip=-1000):
    """Take log and clip to prevent -inf"""
    log_x = np.log(x)
    log_x[log_x==-np.inf] = clip
    return log_x
    
def normalize(x, axis=0):
    """Stable Normalize, prevent division by 0"""
    normalization_vec = x.sum(axis=axis, keepdims=True)
    normalization_vec[normalization_vec==0] = 1e-100
    return x / normalization_vec
    
def exp_normalize(x, axis=0):
    """Numerically stable normalize"""
    x_max = x.max(axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    return normalize(x_exp, axis=axis)

def log_sum_exp(x, axis=None):
    """Version of log(sum(exp(x))) to prevent underflow"""
    x_max = x.max(axis=axis, keepdims=True)
    result = x_max + np.log(np.sum(np.exp(x-x_max), axis=axis, keepdims=True))
    if axis is None:
        return result.item()
    return result