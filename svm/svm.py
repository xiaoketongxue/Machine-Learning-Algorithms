import numpy as np
import math
import random


class Svm:
    def __init__(self, n_iter=100, sigma=10, C=200, toler=0.001):
        self.n_iter = n_iter
        self.sigma = sigma
        self.C = C
        self.toler = toler
        self.b = 0
        self.alpha = None
        self.k = None
        self.supportVecIndex = []
        self.E = None

    def __kernel(self, X):
        n_samples = X.shape[0]

        k = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            xi = X[i, :]
            for j in range(n_samples):
                xj = X[j, :]
                k[i][j] = np.exp(-np.dot(xi - xj, (xi - xj).T) / (2 * np.square(self.sigma)))

        return k

    def __gxi(self, y, i):
        gxi = 0
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        for j in index:
            gxi += self.alpha[j] * y[j] * self.k[j][i]
        gxi += self.b
        return gxi

    def __satisfy_KKT(self, y, i):
        gxi = self.__gxi(y, i)
        yi = y[i]

        if (abs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        elif (abs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) and (
                abs(yi * gxi - 1) < self.toler):
            return True
        return False

    def __Ei(self, y, i):
        gxi = self.__gxi(y, i)
        return gxi - y[i]

    def __get_alpha_j(self, y, E1, i, n_samples):
        E2 = 0
        max_difference = -1
        max_index = -1

        nozero_E_index = [i for i, Ei in enumerate(self.E) if Ei != 0]
        for j in nozero_E_index:
            E2_tmp = self.__Ei(y, j)
            if math.fabs(E1 - E2_tmp) > max_difference:
                max_difference = math.fabs(E1 - E2_tmp)
                E2 = E2_tmp
                max_index = j
        if max_index == -1:
            max_index = i
            while max_index == i:
                max_index = int(random.uniform(0, n_samples))
            E2 = self.__Ei(y, max_index)
        return E2, max_index

    def train(self, X, y):
        n_samples = X.shape[0]

        self.alpha = np.zeros(n_samples)
        self.k = self.__kernel(X)
        self.E = [0 * y[i] for i in range(len(y))]

        parameter_change_sign = True
        for k in range(self.n_iter):
            if parameter_change_sign == False:
                break
            parameter_change_sign = False
            for i in range(n_samples):
                if self.__satisfy_KKT(y, i) == False:
                    E1 = self.__Ei(y, i)
                    E2, j = self.__get_alpha_j(y, E1, i, n_samples)

                    y1 = y[i]
                    y2 = y[j]

                    alpha1_old = self.alpha[i]
                    alpha2_old = self.alpha[j]

                    if y1 != y2:
                        L = max(0, alpha2_old - alpha1_old)
                        H = min(self.C, self.C + alpha2_old - alpha1_old)
                    else:
                        L = max(0, alpha2_old + alpha1_old - self.C)
                        H = min(self.C, alpha2_old + alpha1_old)

                    if L == H:
                        continue

                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]

                    alpha2_new = alpha2_old + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)

                    if alpha2_new < L:
                        alpha2_new = L
                    elif alpha2_new > H:
                        alpha2_new = H

                    alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)

                    b1_new = -1 * E1 - y1 * k11 * (alpha1_new - alpha1_old) \
                             - y2 * k21 * (alpha2_new - alpha2_old) + self.b
                    b2_new = -1 * E2 - y1 * k12 * (alpha1_new - alpha1_old) \
                             - y2 * k22 * (alpha2_new - alpha2_old) + self.b

                    if (alpha1_new > 0) and (alpha1_new < self.C):
                        b_new = b1_new
                    elif (alpha2_new > 0) and (alpha2_new < self.C):
                        b_new = b2_new
                    else:
                        b_new = (b1_new + b2_new) / 2

                    self.alpha[i] = alpha1_new
                    self.alpha[j] = alpha2_new
                    self.b = b_new
                    self.E[i] = self.__Ei(y, i)
                    self.E[j] = self.__Ei(y, j)

                    if abs(alpha2_new - alpha2_old) >= 0.00001:
                        parameter_change_sign = True
        for i in range(n_samples):
            if self.alpha[i] > 0:
                self.supportVecIndex.append(i)

    def predict(self, X, y, x):
        result = 0
        for i in self.supportVecIndex:
            tmp = np.exp(-1 * ((X[i, :] - np.mat(x)) * (X[i, :] - np.mat(x)).T) / (2 * self.sigma ** 2))
            result += self.alpha[i] * y[i] * tmp
        result += self.b
        return np.sign(result)
