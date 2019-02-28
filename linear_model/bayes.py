
class NaiveBayes:
    def __init__(self):
        self.__n_features = None
        self.__prior_count = None
        self.__priori_probability = None
        self.__condition_probability = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        prior_count = dict()
        for i in range(n_samples):
            if y[i] in prior_count:
                prior_count[y[i]] = prior_count.get(y[i]) + 1
            else:
                prior_count[y[i]] = 1

        priori_probability = dict()
        for key in prior_count.keys():
            priori_probability[key] = prior_count.get(key) / n_samples

        condition_probability = dict()
        for j in range(n_features):
            for i in range(n_samples):
                tmp = str(X[i, j]) + '_' + str(y[i]) + '_' + str(j)
                if tmp in condition_probability:
                    condition_probability[tmp] = condition_probability.get(tmp) + 1
                else:
                    condition_probability[tmp] = 1

        self.__n_features = n_features
        self.__prior_count = prior_count
        self.__priori_probability = priori_probability
        self.__condition_probability = condition_probability

    def predict(self, X):
        max_valye = 0.0
        predict_result = None
        for key in self.__priori_probability.keys():
            posterior_probability = float(self.__priori_probability.get(key))
            for j in range(self.__n_features):
                tmp = str(X[j]) + '_' + str(key) + '_' + str(j)
                if tmp in self.__condition_probability:
                    _str = self.__condition_probability.get(tmp)
                    posterior_probability *= float(str(_str)) / float(self.__prior_count.get(key))
                else:
                    posterior_probability *= 1 / float(self.__prior_count.get(key)) + 20
            if posterior_probability > max_valye:
                max_valye = posterior_probability
                predict_result = key

        return predict_result
