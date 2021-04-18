import numpy as np
from ml import kernels


class LinearRegression:
    """Class representing Linear Regression"""

    def fit(self, x, y):
        try:
            self.weights = np.linalg.solve(x.T@x, x.T@y.reshape(-1, 1))
        except:
            self.weights = np.linalg.pinv(x.T@x) @ x.T @ y.reshape(-1, 1)

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


class Perceptron:
    """Rosenblatt's Perceptron"""

    def __init__(self):
        self.weights = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        for xt, yt in zip(x, y):
            y_pred = self.predict(xt)
            if y_pred != yt:
                self.weights += yt * xt

    def predict(self, x):
        return 2*np.heaviside(x@self.weights, 1) - 1

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred==y)


class Winnow:
    """Winnow - takes values in {0, 1}"""

    def __init__(self):
        self.weighs = None
        self.n = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.n = n_features
        self.weights = np.ones(n_features)
        for xt, yt in zip(x, y):
            y_pred = self.predict(xt)
            if y_pred != yt:
                self.weights *= 2.0**((yt-y_pred) * xt)

    def predict(self, x):
        return np.heaviside(x@self.weights - self.n, 1)

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred==y)


class KernelPerceptron:
    """Kernel Perceptron"""

    def __init__(self, kernel, kernel_params=None):
        self.kernel = kernels.get_kernel(kernel_name=kernel,
                                         kernel_params=kernel_params)
        self.alpha = None
        self.x = None

    def fit(self, x, y, epochs=10):
        self.x = x
        n_samples = y.size
        y[y==0] = -1
        self.alpha = np.zeros(n_samples)

        kernel_matrix = self.kernel.compute_matrix(x, x)

        is_change = True
        epoch = 0
        while epoch<epochs and is_change:
            epoch += 1
            is_change = False
            for i in range(n_samples):
                y_pred = 2*np.heaviside(self.alpha @ kernel_matrix[:, i], 1) - 1
                if y_pred != y[i]:
                    self.alpha[i] += y[i]
                    is_change = True

    def update(self, x, y):
        if self.alpha is None:
            self.alpha = np.array([y])
            self.x = x
        else:
            kernel_matrix = np.atleast_1d(self.kernel.compute_matrix(self.x, x))
            y_pred = 2*np.heaviside(self.alpha @ kernel_matrix, 1) - 1
            if y!=y_pred:
                self.alpha = np.append(self.alpha, y)
                self.x = np.vstack((self.x, x))

    def predict(self, x):
        kernel_matrix = self.kernel.compute_matrix(self.x, x)
        projection = self.alpha @ kernel_matrix
        return 2*np.heaviside(projection, 1) - 1

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred==y)


class MulticlassKernelPerceptron:
    """OneVsAll Kernel Perceptron"""

    def __init__(self, kernel, kernel_params=None):
        self.kernel = kernels.get_kernel(kernel_name=kernel,
                                         kernel_params=kernel_params)
        self.alpha = None
        self.x = None

    def fit(self, x, y, epochs=10):
        """Fit to the data one by one"""
        self.x = x
        y = self._encode(y)
        m, n = y.shape
        self.alpha = np.zeros((m, n))

        kernel_matrix = self.kernel.compute_matrix(x, x)

        epoch = 0
        is_change = True
        while epoch < epochs and is_change:
            epoch += 1
            is_change = False
            for i in range(m):
                y_pred = 2*np.heaviside(kernel_matrix[:, i] @ self.alpha, 1) - 1
                mask = y[i]!=y_pred
                if mask.any():
                    self.alpha[i][mask] += y[i][mask]
                    is_change=True

        self.alpha_normalized = self.alpha / np.sqrt(np.sum(self.alpha**2, axis=0))

    def update(self, x, y, n_classes):
        """Used to fit in online setting"""
        y_encoded = -np.ones(n_classes)
        y_encoded[y] = 1

        is_update = False
        if self.alpha is None:
            self.alpha = y_encoded.reshape(1, -1)
            self.x = x.reshape(1, -1)
            is_update = True
        else:
            kernel_matrix = self.kernel.compute_matrix(self.x, x)
            y_pred = 2*np.heaviside(kernel_matrix @ self.alpha, 1) - 1
            mask = y_encoded!=y_pred
            if mask.any():
                new_alpha = np.zeros(self.alpha[0].shape)
                new_alpha[mask] += y_encoded[mask]
                self.alpha = np.vstack((self.alpha, new_alpha))
                self.x = np.vstack((self.x, x))
                is_update=True

        if is_update:
            self.alpha_normalized = self.alpha / np.sqrt(np.sum(self.alpha**2, axis=0))

    def predict(self, x):
        x_train = self.x
        alpha = self.alpha_normalized
        kernel_matrix = self.kernel.compute_matrix(x_train, x)
        y_pred = np.argmax(kernel_matrix.T @ alpha, axis=1)
        return y_pred

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred==y)

    @staticmethod
    def _encode(y):
        y_encoded = -np.ones((y.size, int(y.max()+1)))
        y_encoded[np.arange(y.size), y.astype(int)] = 1
        return y_encoded


class LogisticRegression:
    """Class representing Logistic Regression"""

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, x, y, lr=0.1, tol=1e-4):
        # Initialize params
        if self.w is None:
            self.initialize(x.shape[1])

        # Gradient descent
        prev_loss = np.inf
        h = self.predict(x)
        loss = self.loss(y, h)
        i = 0
        while np.abs(loss-prev_loss) > tol:
            i += 1
            prev_loss = loss
            self.optim_step(lr, x, h, y)
            h = self.predict(x)
            loss = self.loss(y, h)

    def predict(self, x):
        return self.sigmoid(x@self.w + self.b)

    def initialize(self, in_features):
        self.w = 0.1 * np.random.rand(in_features)
        self.b = 0

    def optim_step(self, lr, x, h, y):
        err = h - y
        self.w -= lr * np.mean(err*x.T, axis=1)
        self.b -= lr * np.mean(err)

    @staticmethod
    def sigmoid(x):
        """Stable Sigmoid"""
        result = np.zeros_like(x)

        pos_mask = x >= 0
        result[pos_mask] = 1 / (1+np.exp(-x[pos_mask]))

        neg_mask = ~pos_mask
        exp_x = np.exp(x[neg_mask])
        result[neg_mask] = exp_x / (1+exp_x)

        return result

    @staticmethod
    def loss(y, h):
        """Cross Entropy Loss"""
        eps = 1e-100
        return -np.mean(y*np.log(h+eps) + (1-y)*np.log(1-h+eps))

    def save(self, file_name):
        params = {'w':self.w, 'b':self.b}
        np.savez_compressed(file_name, **params)

    def load(self, file_name):
        params = np.load(file_name)
        self.w = params['w']
        self.b = params['b']
