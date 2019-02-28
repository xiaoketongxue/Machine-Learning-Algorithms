import numpy as np
import math


class Em:
    def __init__(self, n_iter=400):
        self.n_iter = n_iter

    def __Gauss(self, sigma, mu, y):
        return 1 / (np.sqrt(2 * math.pi) * sigma) * np.exp(-(y - mu) * (y - mu) / 2 * (sigma ** 2))

    def __e_step(self, alpha0, alpha1, sigma0, sigma1, mu0, mu1, y):
        gamma0 = alpha0 * self.__Gauss(sigma0, mu0, y)
        gamma1 = alpha1 * self.__Gauss(sigma1, mu1, y)

        gamma = gamma0 + gamma1

        gamma0 = gamma0 / gamma
        gamma1 = gamma1 / gamma

        return gamma0, gamma1

    def __m_step(self, gamma0, gamma1, y):
        mu0 = np.dot(gamma0, y) / np.sum(gamma0)
        mu1 = np.dot(gamma1, y) / np.sum(gamma1)

        newsigma0 = np.sqrt(np.dot(gamma0, (y - mu0) * (y - mu0)) / np.sum(gamma0))
        newsigma1 = np.sqrt(np.dot(gamma1, (y - mu1) * (y - mu1)) / np.sum(gamma1))

        newalpha0 = np.sum(gamma0) / len(gamma0)
        newalpha1 = np.sum(gamma1) / len(gamma1)

        return mu0, mu1, newsigma0, newsigma1, newalpha0, newalpha1

    def fit(self, y):
        alpha0 = 0.5
        mu0 = 0
        sigma0 = 1

        alpha1 = 0.5
        mu1 = 1
        sigma1 = 1

        step = 0
        while step < self.n_iter:
            gamma0, gamma1 = self.__e_step(alpha0, alpha1, sigma0, sigma1, mu0, mu1, y)
            mu0, mu1, sigma0, sigma1, alpha0, alpha1 = self. __m_step(gamma0, gamma1, y)
            step += 1
        return alpha0, alpha1, sigma0, sigma1, mu0, mu1
