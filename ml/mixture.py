import numpy as np
from ml import kernels

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