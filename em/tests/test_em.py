import numpy as np
from em.em import Em
import time


def test_em():
    alpha0 = 0.3
    alpha1 = 0.7
    sigma0 = 0.5
    sigma1 = 1
    mu0 = -2
    mu1 = 0.5
    dataset0 = np.random.normal(loc=mu0, scale=sigma0, size=int(1000000 * alpha0))
    dataset1 = np.random.normal(loc=mu1, scale=sigma1, size=int(1000000 * alpha1))

    dataset = np.array(())
    dataset = np.append(dataset, dataset0)
    dataset = np.append(dataset, dataset1)

    np.random.shuffle(dataset)

    model = Em()
    alpha0, alpha1, sigma0, sigma1, mu0, mu1 = model.fit(dataset)

    print(alpha0, ' ', alpha1, ' ', sigma0, ' ', sigma1, ' ', mu0, ' ', mu1)


if __name__ == '__main__':
    start = time.time()
    test_em()
    end = time.time()
