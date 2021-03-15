import numpy as np


class KMeans:
    """K-Means algorithm with k-means++ initialization"""

    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.means = None
        self.labels = None
        self.n_iters = None

    def fit(self, x):
        self.n_clusters = min(x.shape[0], self.n_clusters)
        self.initialize_means(x)
        means_old = self.means + 1
        self.n_iters = 0
        while not np.allclose(self.means, means_old):
            self.n_iters += 1
            means_old = self.means.copy()
            self.update_labels(x)
            self.update_means(x)

    def initialize_means(self, x):
        n_clusters = self.n_clusters
        n, m = x.shape

        idxs = range(n)
        means = np.zeros((n_clusters, m))
        means[0] = x[np.random.choice(idxs)]
        dists = np.ones(n) * np.inf
        for k in range(1, n_clusters):
            new_dists = np.sum((x - means[k-1])**2, axis=-1)
            dists = np.vstack((dists, new_dists)).min(0)
            dists_sum = dists.sum()
            if dists_sum == 0:
                probs = np.ones(dists.size) / dists.size
            else:
                probs = dists / dists.sum()
            idx = np.random.choice(idxs, p=probs)
            means[k] = x[idx]
        self.means = means

    def update_means(self, x):
        n_clusters = self.n_clusters
        n, m = x.shape
        for k in range(n_clusters):
            k_points = x[self.labels==k]
            if k_points.size:
                self.means[k] = k_points.mean(0)
            else:
                self.means[k] = x[np.random.choice(range(n))]

    def update_labels(self, x):
        x_norm_sq = np.sum(x**2, axis=-1, keepdims=True)
        means_norm_sq = np.sum(self.means**2, axis=-1, keepdims=True)
        norm_sq = x_norm_sq + means_norm_sq.T - 2*(x @ self.means.T)
        self.labels = norm_sq.argmin(-1)
