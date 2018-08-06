import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """X 是 N X D,每行一个样本, Y是一维向量,大小为N"""
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """
        X是N X D, 每行一个样本, 预测分类

        return 分类结果
        """
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_indx = np.argmin(distances)
            Ypred[i] = self.ytr[min_indx]

        return Ypred