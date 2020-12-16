import numpy as np
from joblib import Prallel, delayed
from ml.tree import DecisionTreeClassifier


class RandomForestClassifier(DecisionTreeClassifier):
    """Class representing Random Forest Classifier"""
    
    def __init__(self, n_estimators, max_depth=100,
                 min_samples_split=2, features_size=1.0):
        super().__init__(max_depth, min_samples_split, features_size)
        self.n_estimators = n_estimators
        self.estimators = []
        for _ in range(n_estimators):
            estimator = DecisionTreeClassifier(max_depth,
                                               min_samples_split,
                                               features_size)
            self.estimators.append(estimator)
    
    def fit(self, x, y):
        for estimator in self.estimators:
            x_boot, y_boot = self._get_bootstrap_sample(x, y)
            estimator.fit(x_boot, y_boot)
    
    def predict(self,x):
        n_samples, _ = x.shape
        y_preds = np.zeros((n_samples, self.n_estimators))
        for i, estimator in enumerate(self.estimators):
            y_preds[:, i] = estimator.predict(x)
        y_pred = np.zeros(n_samples)
        for i, yp in enumerate(y_preds):
            y_pred[i] = self._get_most_common(yp)
        return y_pred
    
    @staticmethod
    def _get_bootstrap_sample(x, y):
        n_samples, n_features = x.shape
        boot_idxs = np.random.choice(n_samples, n_samples, replace=True)
        return x[boot_idxs, :], y[boot_idxs]