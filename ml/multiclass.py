import numpy as np
from joblib import Parallel, delayed


class OneVsOneClassifier:
    """One Vs One Multiclass wrapper for binary classifiers"""
    
    def __init__(self, Classifier, **params):
        self.Classifier = Classifier
        self.params = params
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
            delayed(self._fit_single)(x, y, pair, self.Classifier(**self.params))
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
        x_pair, y_pair = OneVsOneClassifier._get_xy_pair(x, y, pair)
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