import numpy as np
from cvxopt import matrix, solvers
from joblib import Parallel, delayed
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
        return np.sign(projection)

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred==y)


class MulticlassSVC:
    """OvO Support Vector Classifier"""

    def __init__(self, kernel='poly', kernel_params={'c':1., 'd':3.}, C=1.0):
        self.kernel = kernels.get_kernel(kernel_name=kernel,
                                         kernel_params=kernel_params)
        self.C = C
        self.n_classes = None
        self.n_classifiers = None
        self.classifiers = []
        self.class_pairs = []

    def fit(self, x, y, n_jobs=-1):
        n_classes = np.unique(y).size
        self.n_classes = n_classes

        pairs = self._get_class_pairs(n_classes)
        self.class_pairs = pairs

        self.n_classifiers = len(pairs)

        self.classifiers = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_single)(x, y, pair, SVC(self.kernel, self.C))
            for pair in pairs
        )

    def predict(self, x):
        y_tmp = np.zeros((x.shape[0], self.n_classifiers))
        for clf, pair in zip(self.classifiers, self.class_pairs):
            y_pred = clf.predict(x)
            mask_pos = y_pred > 0
            mask_neg = ~mask_pos
            y_tmp[mask_pos,pair[0]] += 1
            y_tmp[mask_neg,pair[1]] += 1
        return np.argmax(y_tmp, axis=1)

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred==y)

    @staticmethod
    def _fit_single(x, y, pair, classifier):
        x_pair, y_pair = MulticlassSVC._get_xy_pair(x, y, pair)
        classifier.fit(x_pair, y_pair)
        return classifier

    @staticmethod
    def _get_xy_pair(x, y, pair):
        i, j = pair
        mask_i = y==i
        mask_j = y==j
        mask = mask_i | mask_j

        y_pair = y[mask]
        mask_i = y_pair==i
        mask_j = y_pair==j
        y_pair[mask_i] = 1
        y_pair[mask_j] = -1

        x_pair = x[mask,:]

        return x_pair, y_pair

    @staticmethod
    def _get_class_pairs(n_classes):
        pairs = []
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                pairs.append((i, j))
        return pairs
