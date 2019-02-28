import numpy as np


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, distance='Euclid'):
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.__data = None
        self.__label = None

    def __euclidean_distance(self, vector1, vector2):
        return np.sqrt(np.sum(np.square(vector1 - vector2)))

    def __manhattan_distance(self, vector1, vector2):
        return np.sum(np.abs(vector1 - vector2))

    def fit(self, X, y):
        self.__data = X
        self.__label = y

    def predict(self, X):
        n_samples = self.__data.shape[0]
        dist_list = np.zeros(n_samples)

        for i in range(n_samples):
            xi = self.__data[i]
            if self.distance == 'Euclid':
                cur_dist = self.__euclidean_distance(xi, X)
            elif self.distance == 'Manhattan':
                cur_dist = self.__manhattan_distance(xi, X)
            dist_list[i] = cur_dist

        topK_list = np.argsort(np.array(dist_list))[:self.n_neighbors]

        label_list = [0] * 10
        for index in topK_list:
            label_list[int(self.__label[index])] += 1

        return label_list.index(max(label_list))
