import numpy as np


class Hmm:
    def __init__(self):
        pass

    def forward(self, A, B, PI, O):
        alpha = PI * B[:, O[0]]
        for i in range(len(O) - 1):
            temp = [0] * len(O)
            for j in range(len(A)):
                temp[j] = np.dot(alpha, A[:, j])
            alpha = temp * B[:, O[i + 1]]
        print('前向算法结果：')
        print(np.sum(alpha))

    def vibiter(self, A, B, PI, O):
        alpha = PI * B[:, O[0]]
        index = [[0] * len(A)] * (len(O) - 1)
        for i in range(len(O) - 1):
            temp = [0] * len(O)
            for j in range(len(A)):
                sigma = alpha * A[:, j]
                for k in range(len(sigma)):
                    if sigma[k] > temp[j]:
                        temp[j] = sigma[k]
                        index[i][j] = k
            alpha = temp * B[:, O[i + 1]]
        maxvalue = 0
        maxindex = -1
        for k in range(len(alpha)):
            if alpha[k] > maxvalue:
                maxvalue = alpha[k]
                maxindex = k

        print('维比特算法最优序列：')
        print(maxindex + 1)
        for i in range(len(O) - 1):
            maxindex = index[len(O) - 1 - i - 1][maxindex]
            print(maxindex + 1)



