from numpy import genfromtxt
from tree.tree import DecisionTree
import time


def load_data(file_path):
    data = []
    label = []
    fr = open(file_path)
    for line in fr.readlines():
        current_line = line.strip().split(',')
        data.append([int(int(num) > 128) for num in current_line[1:]])
        label.append(int(current_line[0]))
    return data, label


def test_decision_tree():
    train_data, train_label = load_data('../../datasets/data/mnist_train.csv')
    test_data, test_label = load_data('../../datasets/data/mnist_test.csv')

    model = DecisionTree()
    tree = model.create_tree(train_data, train_label)

    m = len(test_data)
    error_count = 0
    for i in range(m):
        if test_label[i] != model.predict(test_data[i], tree):
            error_count += 1

    return 1 - error_count / m


if __name__ == '__main__':
    start = time.time()
    acc_rate = test_decision_tree()
    end = time.time()
    print('accuracy rate is:', acc_rate)
    print('time span:', end - start)
