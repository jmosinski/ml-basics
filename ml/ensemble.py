import numpy as np
from joblib import Parallel, delayed
from ml.tree import DecisionTreeClassifier


class RandomForestClassifier(DecisionTreeClassifier):
    """Class representing Random Forest Classifier"""
    
    def __init__(self, n_estimators, max_depth=100,
                 min_samples_split=2, max_features=1.0):
        super().__init__(max_depth, min_samples_split, max_features)
        self.n_estimators = n_estimators
        self.estimators = []
        for i in range(self.n_estimators):
            estimator = DecisionTreeClassifier(self.max_depth,
                                               self.min_samples_split,
                                               self.max_features) 
            self.estimators.append(estimator)

    def fit(self, x, y, n_jobs=-1):
        self.estimators = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_single)(x, y, self.estimators[i])
            for i in range(self.n_estimators)
        )
        
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
    def _fit_single(x, y, estimator):
        # Get bootstrap sample
        n_samples, n_features = x.shape
        boot_idxs = np.random.choice(n_samples, n_samples, replace=True)
        x_boot, y_boot = x[boot_idxs, :], y[boot_idxs]
        # Fit
        estimator.fit(x_boot, y_boot)
        return estimator