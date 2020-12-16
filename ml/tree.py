import numpy as np


class Node:
    """Class representing tree node"""
    
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None
    
    
class DecisionTreeClassifier:
    """Class representing Decision Tree Classifier"""
    
    def __init__(self, max_depth=100, min_samples_split=2, features_size=1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features_size = features_size
        self.root = None
        
    def fit(self, x, y):
        self.root = self._grow_tree(x, y, depth=0)
        
    def predict(self, x):
        y_pred = np.zeros(x.shape[0], dtype=int)
        for i, xi in enumerate(x):
            value = self._traverse_tree(xi, self.root)
            y_pred[i] = value
        return y_pred
    
    def score(self, x, y):
        y_pred = self.predict(x)
        is_correct = y_pred==y
        acc = is_correct.mean()
        return acc
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        else:
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        
    def _grow_tree(self, x, y, depth):
        n_samples, n_features = x.shape
        n_classes = len(set(y))
        node = None
        
        # Stop growth criteria
        if (depth >= self.max_depth 
            or n_classes <= 1
            or n_samples <= self.min_samples_split):
            node = Node(value=self._get_most_common(y))
        else:
            feature, thresh = self._get_best_split(x, y)
            if feature is not None:
                # Split data
                row_mask_left = x[:,feature]<=thresh
                row_mask_right = x[:,feature]>thresh
                x_left, x_right = x[row_mask_left,:], x[row_mask_right,:]
                y_left, y_right = y[row_mask_left], y[row_mask_right]
                # Create new node
                node = Node(feature=feature, threshold=thresh)
                node.left = self._grow_tree(x_left, y_left, depth+1)
                node.right = self._grow_tree(x_right, y_right, depth+1)
            else:
                node = Node(value=self._get_most_common(y))
        return node

    def _get_best_split(self, x, y):
        n_samples, n_features = x.shape
        best_feature, best_thresh = None, None
        best_impurity = np.inf
        features = np.random.choice(n_features, 
                                    int(np.ceil(self.features_size*n_features)), 
                                    replace=False)
        for feature in features:
            x_feature = x[:, feature]
            threshes = self._get_potential_thresholds(x_feature)
            if threshes is None: continue
            for thresh in threshes:
                impurity = self._get_gini_impurity(x_feature, y, thresh)
                if impurity <= best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_thresh = thresh
        return best_feature, best_thresh

    @staticmethod
    def _get_potential_thresholds(x):
        thresholds = None
        x_len = x.size
        x = np.sort(x)
        if x_len > 50:
            quantile_idxs = (np.arange(0, 1.02, 0.02) * (x_len-1)).astype(int)
            thresholds = np.unique(x[quantile_idxs])
        else:
            thresholds = np.unique(x)
            
        if thresholds.size >= 2:
            return (thresholds[:-1] + thresholds[1:]) / 2
        else:
            return None
    
    @staticmethod
    def _get_most_common(y):
        y_unique, y_counts = np.unique(y, return_counts=True)
        return y_unique[y_counts.argmax()]
    
    @staticmethod
    def _split(x, y, feature, threshold):
        row_mask_left = x[:feature]<=threshold
        row_mask_right = x[:feature]>threshold
        x_left, x_right = x[row_mask_left,:], x[row_mask_right,:]
        y_left, y_right = y[row_mask_left], y[row_mask_right]
        return x_left, x_right, y_left, y_right
    
    @staticmethod
    def _get_gini_impurity(x, y, threshold):
        yl, yr = y[x<=threshold], y[x>threshold]
        y_len, yl_len, yr_len = y.size, yl.size, yr.size
        _, yl_counts = np.unique(yl, return_counts=True)
        _, yr_counts = np.unique(yr, return_counts=True)
        yl_probs = yl_counts / yl_len
        yr_probs = yr_counts / yr_len
        gini_l = 1 - yl_probs @ yl_probs
        gini_r = 1 - yr_probs @ yr_probs
        return (yl_len*gini_l + yr_len*gini_r) / y_len