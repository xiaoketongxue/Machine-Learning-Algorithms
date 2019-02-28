from linear_model.perceptron import Perceptron
from numpy import genfromtxt
import time


def test_perceptron_accuracy():
    train = genfromtxt('../../datasets/data/mnist_train.csv', delimiter=',')
    test = genfromtxt('../../datasets/data/mnist_test.csv', delimiter=',')

    train_data = train[:, 1:(train.shape[1] - 1)]
    train_label = train[:, 0]
    train_label[train_label < 5] = -1
    train_label[train_label >= 5] = 1

    model = Perceptron()
    model.fit(train_data, train_label)

    test_data = test[:, 1:(test.shape[1] - 1)]
    test_label = test[:, 0]
    test_label[test_label < 5] = -1
    test_label[test_label >= 5] = 1

    m = test_data.shape[0]
    error_count = 0
    for i in range(m):
        xi = test_data[i]
        yi = test_label[i]
        if yi * model.predict(xi) <= 0:
            error_count += 1

    return 1 - error_count / m


if __name__ == '__main__':
    start = time.time()
    acc_rate = test_perceptron_accuracy()
    end = time.time()
    print('accuracy rate is:', acc_rate)
    print('time span:', end - start)

