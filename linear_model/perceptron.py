import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.0001, n_iter=50):
        self.w = None
        self.b = None
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = np.zeros(1)

        for k in range(self.n_iter):
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                if yi * (np.dot(self.w, xi.T) + self.b) <= 0:
                    self.w += self.learning_rate * yi * xi
                    self.b += self.learning_rate * yi

    def predict(self, X):
        if np.sign(np.dot(self.w, X.T) + self.b) >= 0:
            return 1
        else:
            return -1
