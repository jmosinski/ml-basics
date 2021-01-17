import numpy as np
from cvxopt import matrix, solvers
from ml import kernels


class SVC:
    """Support Vector Classifier optimized with Quadratic Programming"""

    def __init__(self, kernel='poly', kernel_params={'c':1., 'd':3.}, C=1.0):
        self.kernel = kernels.get_kernel(kernel_name=kernel,
                                         kernel_params=kernel_params)
        self.C = C
        self.dual_coef = None
        self.support_vectors = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        kernel_matrix = self.kernel.compute_matrix(x, x)

        # Set up QP objective
        P = matrix(np.outer(y, y)*kernel_matrix)
        q = matrix(-np.ones((n_samples, 1)))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), self.C*np.ones(n_samples))))
        A = matrix(y, (1, n_samples), 'd')
        b = matrix(np.zeros(1))
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x']).reshape(-1)

        # Save only params for support vectors
        sv_mask = alphas > 1e-3 * alphas.max()
        self.dual_coef = alphas[sv_mask] * y[sv_mask]
        self.support_vectors = x[sv_mask,:]
        self.intercept = np.mean(y[sv_mask] - self.dual_coef * kernel_matrix[sv_mask, sv_mask])

    def predict(self, x):
        kernel_matrix = self.kernel.compute_matrix(self.support_vectors, x)
        projection = (self.dual_coef @ kernel_matrix) + self.intercept
        return 2*np.heaviside(projection, 1) - 1

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred==y)
