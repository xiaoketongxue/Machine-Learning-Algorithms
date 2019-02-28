import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.001,n_iter=200):
        self.w = None
        self.b = None
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        n_samples, n_features = X.shape

        new_X = np.zeros((n_samples, n_features+1))
        for i in range(n_samples):
            new_X[i] = np.append(X[i], 1)

        self.w = np.zeros(n_features+1)

        for i in range(self.n_iter):
            for j in range(n_samples):
                xi = new_X[j]
                yi = y[j]
                self.w -= self.learning_rate * (xi * yi - (np.exp(np.dot(self.w, xi)) * xi) / (1 + np.exp(np.dot(self.w,xi))))

    def predict(self, X):
        new_X = np.append(X, 1)

        if np.exp(np.dot(self.w, new_X)) / (1 + np.exp(np.dot(self.w, new_X))) >= 0.5:
            return 1
        else:
            return 0