import numpy as np
from ml import kernels

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
    
    
class MulticlassKernelPerceptron:
    """Multi-class Kernel Perceptron"""
    
    def __init__(self, kernel='linear', kernel_params=None):
        self.kernel = kernels.get_kernel(kernel_name=kernel,
                                         kernel_params=kernel_params)
        self.history = {'train_error_rate': []}
        self.alpha = None
        self.x = None
        
    def fit(self, x, y, epochs=1):
        self.x = x
        y = self._encode(y)
        m, n = y.shape
        alpha = np.zeros((m, n))
            
        kernel_matrix = self.kernel.compute_matrix(x, x)

        epoch = 0
        is_change = True
        while epoch < epochs and is_change:
            epoch += 1
            is_change = False
            for i in range(m):
                y_pred = np.sign(kernel_matrix[:, i] @ alpha)
                mask = y[i]!=y_pred
                if mask.any():
                    alpha[i][mask] += y[i][mask]
                    is_change=True
         
        self.alpha = alpha
        self.alpha_normalized = alpha / np.sqrt(np.sum(alpha**2, axis=0))
    
    def predict(self, x):
        x_train = self.x
        alpha = self.alpha_normalized
        kernel_matrix = self.kernel.compute_matrix(x_train, x)
        y_pred = np.argmax(kernel_matrix.T @ alpha, axis=1)
        return y_pred
    
    def score(self, x, y):
        y_pred = self.predict(x)
        is_correct = y_pred==y
        acc = is_correct.mean()
        return acc
    
    @staticmethod
    def _encode(x):
        x_encoded = -np.ones((x.shape[0], int(x.max()+1)))
        x_encoded[np.arange(x.size), x] = 1
        return x_encoded