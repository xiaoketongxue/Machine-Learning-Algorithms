from numpy import genfromtxt
from neighbors.classification import KNeighborsClassifier
import time


def test_KNeighborsClassifier():
    train = genfromtxt('../../datasets/data/mnist_train.csv', delimiter=',')
    test = genfromtxt('../../datasets/data/mnist_test.csv', delimiter=',')
    train_data = train[:, 1:(train.shape[1] - 1)]
    train_label = train[:, 0]

    model = KNeighborsClassifier(5,'Manhattan')
    model.fit(train_data, train_label)

    test_data = test[:, 1:(test.shape[1] - 1)]
    test_label = test[:, 0]

    m = 200
    error_count = 0
    for i in range(m):
        yi = test_label[i]
        if model.predict(test_data[i, :]) != yi:
            error_count += 1

    return 1 - error_count / m


if __name__ == '__main__':
    start = time.time()
    acc_rate = test_KNeighborsClassifier()
    end = time.time()
    print('accuracy rate is:', acc_rate)
    print('time span:', end - start)
