from numpy import genfromtxt
import time
from ensemble.adaboost import AdaBoost
import numpy as np


def test_adaboost_accuracy():
    train = genfromtxt('../../datasets/data/mnist_train.csv', delimiter=',')
    test = genfromtxt('../../datasets/data/mnist_test.csv', delimiter=',')

    train_data = train[:1000, 1:(train.shape[1] - 1)]
    train_data[train_data < 128] = 0
    train_data[train_data >= 128] = 1

    train_label = train[:1000, 0]
    train_label[train_label >= 1] = -1
    train_label[train_label == 0] = 1

    model = AdaBoost()
    tree = model.fit(train_data, train_label)

    test_data = test[:100, 1:(test.shape[1] - 1)]
    test_data[test_data < 128] = 0
    test_data[test_data >= 128] = 1
    test_label = test[:100, 0]
    test_label[test_label >= 1] = -1
    test_label[test_label == 0] = 1

    m = test_data.shape[0]
    error_count = 0
    for i in range(m):
        result = 0
        print(i)
        for current_tree in tree:
            div = current_tree['div']
            rule = current_tree['rule']
            feature = current_tree['feature']
            alpha = current_tree['alpha']
            result += alpha * model.predict(test_data[i], div, rule, feature)
        if np.sign(result) != test_label[i]:
            error_count += 1

    return 1 - error_count / m


if __name__ == '__main__':
    start = time.time()
    acc_rate = test_adaboost_accuracy()
    end = time.time()
    print('accuracy rate is:', acc_rate)
    print('time span:', end - start)
