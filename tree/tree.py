import numpy as np


class DecisionTree:
    def __init__(self):
        pass

    def __experience_entropy(self, y):
        experience_entropy = 0
        label_set = set([label for label in y])
        for i in label_set:
            p = y[y == i].size / y.size
            experience_entropy += -1 * p * np.log2(p)

        return experience_entropy

    def __condition_entropy(self, X, y):

        condition_entropy = 0
        train_data_set = set([data for data in X])
        for i in train_data_set:
            condition_entropy += X[X == i].size / X.size * self.__experience_entropy(y[X == i])

        return condition_entropy

    def __best_feature(self, X_list, y_list):
        X = np.array(X_list)
        y = np.array(y_list)
        n_features = X.shape[1]
        max_information_gain = 0
        best_feature = 0

        for feature in range(n_features):
            experience_entropy = self.__experience_entropy(y)
            information_gain = experience_entropy - self.__condition_entropy(np.array(X[:, feature].flat), y)

            if information_gain > max_information_gain:
                max_information_gain = information_gain
                best_feature = feature
        return best_feature, max_information_gain

    def __majority_class(self, y):

        class_count = {}
        for i in range(len(y)):
            if y[i] in class_count.keys():
                class_count[y[i]] += 1
            else:
                class_count[y[i]] = 1

        class_sort = sorted(class_count.items(), key=lambda x: x[1], reverse=True)

        return class_sort[0][0]

    def __sub_data(self, X, y, A, a):
        sub_x = []
        sub_y = []
        for i in range(len(X)):
            if X[i][A] == a:
                sub_x.append(X[i][0:A]+X[i][A + 1:])
                sub_y.append(y[i])

        return sub_x, sub_y

    def create_tree(self, X, y):

        label_set = set([label for label in y])
        if len(label_set) == 1:
            return y[0]

        if len(X[0]) == 0:
            return self.__majority_class(y)

        Ag, max_information_gain = self.__best_feature(X, y)

        epsilon = 0.1
        if max_information_gain < epsilon:
            return self.__majority_class(y)

        tree_dict = {Ag: {}}

        sub_x, sub_y = self.__sub_data(X, y, Ag, 0)
        tree_dict[Ag][0] = self.create_tree(sub_x, sub_y)
        sub_x, sub_y = self.__sub_data(X, y, Ag, 1)
        tree_dict[Ag][1] = self.create_tree(sub_x, sub_y)

        return tree_dict

    def predict(self, X, tree):
        while True:
            (key, value), = tree.items()
            if type(tree[key]).__name__ == 'dict':
                data_value = X[key]
                del X[key]
                tree = value[data_value]
                if type(tree).__name__ == 'int':
                    return tree
            else:
                return value

