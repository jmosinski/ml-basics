import numpy as np
from ml import kernels
from utils import log_clip, log_sum_exp
from utils import normalize, exp_normalize


#################
# Linear Models #
#################

class LinearRegression:
    """Class representing Linear Regression"""

    def fit(self, x, y):
        try:
          self.weights = np.linalg.solve(x.T@x, x.T@y.reshape(-1, 1))
          return True
        except:
          # Let callers know if singluar matrix
          return None

    def predict(self, x):
        return x @ self.weights


class KernelRidgeRegression:
    """Class representing Kernel Ridge Regression"""

    def __init__(self, gamma, kernel='linear', kernel_params=None):
        self.gamma = gamma
        self.alpha = None
        self.kernel = kernels.get_kernel(kernel_name=kernel,
                                         kernel_params=kernel_params)

    def fit(self, x, y):
        kernel = self.kernel.compute_matrix(x, x)
        m = x.shape[0]
        gamma = self.gamma
        alpha = np.linalg.solve(kernel + gamma*m*np.identity(m), y.reshape(-1, 1))
        self.alpha = alpha
        self.x = x

    def predict(self, x):
        kernel = self.kernel.compute_matrix(self.x, x)
        return kernel.T @ self.alpha

    
###################    
# Gaussian Models #
###################

class GaussianMixture:
    """Class representing Mixture of Gaussians model."""

    def __init__(self, n_components):
        """Initializes number of mixture components."""
        self.params = {'n_components': n_components}
        self.history = {'train_loglike': []}

    def fit(self, x, max_iter=10, tol=1e-3, verbose=True):
        """Estimate model parameters with the EM algorithm."""
        # Initialize parameters
        m, n = data.shape
        params = self.params
        n_components = params['n_components']
        if len(params) == 1:
            params['dims'] = n
            params['weight'] = (1 / n_components) * np.ones(shape=(n_components))
            params['mean'] = np.zeros((n_components, n))
            params['cov'] = np.zeros((n_components, n, n))
            params['cov_inv'] = np.zeros((n_components, n, n))
            params['cov_det'] = np.zeros((n_components))
            for comp in range(n_components):
                params['mean'][comp] = data[int(np.random.rand()*n-1)]
                cov = np.cov(data.T) * (1.0 + 0.1*np.random.uniform()*np.eye(n))
                params['cov'][comp] = cov
                params['cov_inv'][comp] = np.linalg.inv(cov)
                params['cov_det'][comp] = np.linalg.det(cov)

        self.history['train_loglike'].append(self.get_loglike(data))
        post_hidden = np.zeros(shape=(n_components, m))
        # Fit
        iters = 0
        like_diff = tol+1
        while iters <= max_iter and like_diff > tol:
            # E-Step
            for comp in range(n_components):
                weight = self.params['weight'][comp]
                post_hidden[comp] = weight * self.component_prob(data, comp)
            post_hidden = post_hidden / post_hidden.sum(axis=0)

            # M-Step
            for comp in range(n_components):
                new_weight = np.sum(post_hidden[comp]) / post_hidden.sum()

                new_mean = (np.sum(post_hidden[comp] * data.T, axis=1)
                           / post_hidden[comp].sum())

                new_cov = ((post_hidden[comp] * (data-new_mean).T)
                           @ (data-new_mean) / np.sum(post_hidden[comp]))

                # Update
                params['weight'][comp] = new_weight
                params['mean'][comp] = new_mean
                params['cov'][comp] = new_cov
                params['cov_inv'][comp] = np.linalg.inv(new_cov)
                params['cov_det'][comp] = np.linalg.det(new_cov)
                self.params = params

            # Calculate log-likelihood and print
            loglike = self.get_loglike(data)
            if verbose:
                print('Log Likelihood After Iter {} : {:4.3f}\n'.format(iters, loglike))
            iters += 1
            self.history['train_loglike'].append(loglike)
            like_diff = (self.history['train_loglike'][-1]
                         - self.history['train_loglike'][-2])

    def component_prob(self, x, component):
        """PDF of a component."""
        mean = self.params['mean'][component]
        cov = self.params['cov'][component]
        cov_inv = self.params['cov_inv'][component]
        cov_det = self.params['cov_det'][component]
        dims = self.params['dims']

        diffs = x - mean
        if len(diffs.shape) == 1:
            diffs = diffs.reshape(1, -1)
        probs = (np.exp(-0.5 * np.sum(diffs @ cov_inv * diffs, axis=1))
                / np.sqrt((2*np.pi)**(dims) * cov_det))
        return probs

    def component_logprob(self, x, component):
        """Log PDF of a component."""
        mean = self.params['mean'][component]
        cov = self.params['cov'][component]
        cov_inv = self.params['cov_inv'][component]
        cov_det = self.params['cov_det'][component]
        dims = self.params['dims']

        diffs = x - mean
        if len(diffs.shape) == 1:
            diffs = diffs.reshape(1, -1)
        const = np.log(cov_det) + dims*np.log(2*np.pi)
        logprobs = -0.5 *(np.sum(diffs @ cov_inv * diffs, axis=1) + const)
        return logprobs

    def predict_proba(self, x):
        """Predict posterior probability of the mixture given the data."""
        n_components = self.params['n_components']
        prob = 0
        for comp in range(n_components):
            weight = self.params['weight'][comp]
            prob += weight * self.component_prob(x, comp)
        return prob

    def predict(self, x, thresh=0.5):
        """Predict labels given the threshold"""
        probs = self.predict_proba(x)
        preds = np.floor(probs + 1.0 - thresh)
        return preds
    
    def get_loglike(self, x):
        n_comp = self.params['n_components'])
        like = np.zeros(data.shape[0])
        for comp in range(n_comp):
            prob = self.component_prob(x, comp)
            weight = self.params['weight'][comp]
            like += weight * prob
        return np.sum(np.log(like))
        
###############
# Tree Models #
###############

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


####################
# Graphical Models #
####################

"""Not finished
- add steady states from eigs
- generalize to all orders
- add simulate
"""
class MarcovChain:
    """Class representing 1st order Marcov Chain"""

    def __init__(self, n_states):
        self.n_states = n_states
        self.transition_matrix = np.ones((n_states, n_states)) / n_states
        self.state_probs = np.ones(n_states) / n_states

    def fit(self, data):
        n_states = self.n_states
        # Get Maximum Likelihood estimation of transiton matrix
        transition_matrix = np.zeros((n_states, n_states))
        for seq in data:
            for i in range(1, len(seq)):
                transition_matrix[seq[i], seq[i-1]] += 1
        transition_matrix /= transition_matrix.sum(axis=0)
        self.transition_matrix = transition_matrix
        # Get ML estimate of state probs
        _, state_probs = np.unique(data, return_counts=True)
        self.state_probs = state_probs / state_probs.sum()

    def describe(self):
        print('Transition matrix:')
        print(self.transition_matrix)
        print()
        print('State probs:')
        print(self.state_probs)

        
class MarkovMixture:
    """Class representing Mixture of Markov Chains"""
    
    def __init__(self, n_components, n_states):
        self.params = {
            'n_components': n_components,
            'n_states': n_states,
        }
        self.history = {'train_loglike': []}

    def fit(self, x, max_iter=1):
        # Initialize parameters
        params = self.params
        n_comp = params['n_components']
        n_states = params['n_states']
        
        transition_matrix = normalize(np.random.rand(n_comp, n_states, n_states), axis=1)
        self.params['transition_matrix'] = transition_matrix

        init_probs = normalize(np.random.rand(n_comp, n_states), axis=1)
        self.params['initial_probs'] = init_probs
        
        n_seq, n_t = x.shape
        comp_post = normalize(np.random.rand(n_comp, n_seq))
                               
        comp_probs = normalize(np.random.rand(n_comp))
        self.params['component_probs'] = comp_probs
        
        transition_counts = np.zeros((n_seq, n_states, n_states))
        init_states_dummy = np.zeros((n_seq, n_states))
        for n, seq in enumerate(x):
            init_states_dummy[n, seq[0]] = 1
            for i in range(1, n_t):
                transition_counts[n, seq[i], seq[i-1]] += 1
        transition_counts = transition_counts.reshape(-1, n_states**2)
        
        # Fit
        iters = 0
        for i in range(max_iter):
            log_transition_matrix = log_clip(transition_matrix).reshape(n_comp, -1)
            log_init_probs = log_clip(init_probs)
            log_comp_probs = log_clip(comp_probs)
            
            # E-Step
            comp_loglikes = ((log_init_probs @ init_states_dummy.T)
                             + (log_transition_matrix @ transition_counts.T)
                             + log_comp_probs.reshape(-1, 1))
            
            comp_post = exp_normalize(comp_loglikes)
            self.history['train_loglike'].append(log_sum_exp(comp_loglikes))
            
            # M-Step
            init_probs = normalize(comp_post @ init_states_dummy, axis=1)
            
            transition_matrix = normalize((comp_post @ transition_counts).reshape(-1, n_states, n_states), axis=1)
            
            comp_probs = normalize(comp_post.sum(axis=1, keepdims=True)).reshape(-1)
        
        # Update
        self.params['transition_matrix'] = transition_matrix
        self.params['initial_probs'] = init_probs
        self.params['component_probs'] = comp_probs
    
    def get_comp_posterior(self, x):
        return exp_normalize(self.get_comp_loglike(x))
    
    def get_loglike(self, x):
        return log_sum_exp(self.get_comp_loglike(x))
    
    def get_comp_loglike(self, x):
        n_comp = self.params['n_components']
        n_states = self.params['n_states']
        n_seq, n_t = x.shape
        
        transition_counts = np.zeros((n_seq, n_states, n_states))
        init_states_dummy = np.zeros((n_seq, n_states))
        for n, seq in enumerate(x):
            init_states_dummy[n, seq[0]] = 1
            for i in range(1, n_t):
                transition_counts[n, seq[i], seq[i-1]] += 1
        transition_counts = transition_counts.reshape(-1, n_states**2)

        transition_matrix = self.params['transition_matrix']
        init_probs = self.params['initial_probs']
        comp_probs = self.params['component_probs']
        log_transition_matrix = log_clip(transition_matrix).reshape(n_comp, -1)
        log_init_probs = log_clip(init_probs)
        log_comp_probs = log_clip(comp_probs)
        
        comp_loglikes = ((log_init_probs @ init_states_dummy.T)
                          + (log_transition_matrix @ transition_counts.T)
                          + log_comp_probs.reshape(-1, 1))
        return comp_loglikes