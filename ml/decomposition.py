import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.means = None
        self.components = None
        self.singular_values = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, x):
        self.means = x.mean(0)
        u, s, vh = np.linalg.svd(x-self.means, full_matrices=False)
        self.components = vh[:self.n_components].T
        self.singular_values = s[:self.n_components]
        self.explained_variance_ratio = self.singular_values**2 / np.sum(s**2)

    def transform(self, x):
        return (x-self.means) @ self.components

    def inv_transform(self, x):
        return x @ self.components.T + self.means
