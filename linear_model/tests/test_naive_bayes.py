from numpy import genfromtxt
import time
import numpy as np
from linear_model.bayes import NaiveBayes


def test_naive_bayes():
    train = genfromtxt('../../datasets/data/mnist_train.csv', delimiter=',')
    test = genfromtxt('../../datasets/data/mnist_test.csv', delimiter=',')
    train_data = train[:, 1:(train.shape[1] - 1)]
    train_data[train_data < 128] = 0
    train_data[train_data >= 128] = 1

    train_label = train[:, 0]

    model = NaiveBayes()
    model.fit(train_data, train_label)

    test_data = test[:, 1:(test.shape[1] - 1)]
    test_data[test_data < 128] = 0
    test_data[test_data >= 128] = 1
    test_label = test[:, 0]

    m = test_data.shape[0]
    error_count = 0
    for i in range(m):
        if np.abs(model.predict(test_data[i]) - test_label[i]) > 0.0000001:
            error_count += 1

    return 1 - error_count / m


if __name__ == '__main__':
    start = time.time()
    acc_rate = test_naive_bayes()
    end = time.time()
    print('accuracy rate is:', acc_rate)
    print('time span:', end - start)
