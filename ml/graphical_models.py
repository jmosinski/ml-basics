import numpy as np
from utils import log_clip, log_sum_exp
from utils import normalize, exp_normalize


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