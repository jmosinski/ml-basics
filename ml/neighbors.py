import numpy as np


class KNeighborsClassifier:
    """K-Nearest Neighbors Classfier"""

    def __init__(self, k=1):
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        norm_sq = self._norm_sq(self.x, x)
        knn_idx = np.argsort(norm_sq, axis=0)[:self.k]
        knn = self.y[knn_idx]
        y_pred = [self._most_common(knn_i) for knn_i in knn.T]
        return np.array(y_pred)

    @staticmethod
    def _norm_sq(x, t):
        x_norm_sq = np.sum(x**2, axis=-1, keepdims=True)
        t_norm_sq = np.sum(t**2, axis=-1, keepdims=True)
        norm_sq = x_norm_sq + t_norm_sq.T - 2*(x @ t.T)
        return norm_sq

    @staticmethod
    def _most_common(y):
        return np.bincount(y).argmax()

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred==y)
