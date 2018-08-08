import numpy as np

class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """X 是 N X D,每行一个样本, Y是一维向量,大小为N"""
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        X是N X D, 每行一个样本, 预测分类

        return 分类结果
        """
        if num_loops == 0:
            dists = self.compute_distance_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distance_one_loops(X)
        elif num_loops == 2:
            dists = self.compute_distance_two_loops(X)
        else:
            raise ValueError('错误的值num_loops = %d' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distance_no_loops(self, X):
        """
        传入测试样本X
        返回每个测试样本和每个训练样本的距离（num_test, num_train）
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists += np.sum(self.X_train ** 2, axis=1).reshape(1, num_train)
        dists -= np.sum(X ** 2, axis=1).reshape(num_test, 1)
        dists = np.sqrt(dists)

        return dists

    def compute_distance_one_loops(self, X):
        """
        传入测试样本X
        返回每个测试样本和每个训练样本的距离（num_test, num_train）
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sqrt(np.sum((self.X_train - X[i]) ** 2, 1))

        return dists

    def compute_distance_two_loops(self, X):
        """
        传入测试样本X
        返回每个测试样本和每个训练样本的距离（num_test, num_train）
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))

        return dists
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.y_train[np.argsort(dists[i])[0:k]]
            y_pred[i] = np.bincount(closest_y).argmax()

        return y_pred
