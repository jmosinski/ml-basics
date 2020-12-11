import numpy as np


class Kernel:
    """Class specifying object of Kernel type"""

    def __init__(self, params=None):
        self.params = params

    def apply(self, x, t):
        pass

    def compute_matrix(self, x, t):
        m_x = x.shape[0]
        m_t = t.shape[0]
        kernel_matrix = np.zeros((m_x, m_t))
        for i in range(m_x):
            for j in range(m_t):
                kernel_matrix[i, j] = self.apply(x[i], t[j])
        return kernel_matrix


class Linear(Kernel):
    """Linear kernel class, K(x, t) = dot(x, t)"""

    def apply(self, x, t):
        return np.dot(x, t)

    def compute_matrix(self, x, t):
        return x @ t.T


class Polynomial(Kernel):
    """Polynomial kernel class, K(x, t) = (dot(x, t) + c)^d"""

    def apply(self, x, t):
        c = self.params['c']
        d = self.params['d']
        return (np.dot(x, t) + c)**d

    def compute_matrix(self, x, t):
        c = self.params['c']
        d = self.params['d']
        return (x@t.T + c)**d


class RBF(Kernel):
    """RBF kernel class, K(x, t) = exp(-||x-t||^2 / (2*sigma^2))"""

    def apply(self, x, t):
        sigma = self.params['sigma']
        two_sigma_sq = 2 * sigma**2
        diff = x - t
        return np.exp(-np.dot(diff, diff) / two_sigma_sq)

    def compute_matrix(self, x, t):
        two_sigma_sq = 2 * self.params['sigma']**2
        x_norm_sq = np.sum(x**2, axis=-1, keepdims=True)
        t_norm_sq = np.sum(t**2, axis=-1, keepdims=True)
        norm_sq = x_norm_sq + t_norm_sq.T - 2*x@t.T
        return np.exp(-norm_sq / two_sigma_sq)


def get_kernel(kernel_name, kernel_params=None):
    """Returns specified kernel object"""
    kernel_name = kernel_name.lower()
    if kernel_name == 'linear':
        return Linear()
    elif kernel_name == 'polynomial':
        return Polynomial(kernel_params)
    elif kernel_name == 'rbf':
        return RBF(kernel_params)
    else:
        return None
