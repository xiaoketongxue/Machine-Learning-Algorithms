import numpy as np


class AdaBoost:
    def __init__(self, tree_num=50):
        self.tree_num = tree_num

    def __gx_error(self, X, y, i, div, rule, D):
        n_samples, n_features = X.shape
        e = 0
        x = X[:, i]
        predict = np.array(())
        if rule == 'LisOne':
            L = 1
            H = -1
        else:
            L = -1
            H = 1
        for j in range(n_samples):
            if x[j] < div:
                predict = np.append(predict, L)
                if y[j] != L:
                    e += D[j]
            elif x[j] >= div:
                predict = np.append(predict, H)
                if y[j] != H:
                    e += D[j]

        return predict, e

    def __create_boost_tree(self, X, y, D):
        n_samples, n_features = X.shape

        BoostTree = {}
        BoostTree['e'] = 1

        for i in range(n_features):
            for div in [-0.5, 0.5, 1.5]:
                for rule in ['LisOne', 'HisOne']:
                    Gx, e = self.__gx_error(X, y, i, div, rule, D)
                    if e < BoostTree['e']:
                        BoostTree['e'] = e
                        BoostTree['div'] = div
                        BoostTree['rule'] = rule
                        BoostTree['Gx'] = Gx
                        BoostTree['feature'] = i

        return BoostTree

    def fit(self, X, y):
        n_samples, n_features = X.shape
        D = [1 / n_samples] * n_samples
        tree = []

        finalpredict = [0] * n_samples
        for i in range(self.tree_num):
            current_tree = self.__create_boost_tree(X, y, D)
            alpha = 1 / 2 * np.log((1 - current_tree['e']) / current_tree['e'])
            Gx = current_tree['Gx']
            D = np.multiply(D, np.exp(-1 * alpha * np.multiply(y, Gx))) / sum(D)
            current_tree['alpha'] = alpha
            tree.append(current_tree)

            finalpredict += alpha * Gx
            error = sum([1 for i in range(n_samples) if np.sign(finalpredict[i]) != y[i]])
            print('iter:%d' % (i))
            error_rate = error / n_samples

            if error_rate == 0:
                return tree
        return tree

    def predict(self, x, div, rule, feature):
        if rule == 'LisOne':
            L = 1
            H = -1
        else:
            L = -1
            H = 1

        if x[feature] < div:
            return L
        else:
            return H


