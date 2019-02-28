import numpy as np
from hmm.hmm import Hmm
import time


def test_hmm():
    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    PI = np.array([0.2, 0.4, 0.4])
    O = np.array([0, 1, 0])

    model = Hmm()
    model.forward(A, B, PI, O)
    model.vibiter(A, B, PI, O)


if __name__ == '__main__':
    start = time.time()
    test_hmm()
    end = time.time()
