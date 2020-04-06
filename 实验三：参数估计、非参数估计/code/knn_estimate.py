import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    with open(filename, encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        _data = [line for line in csv_reader]
    data = np.array(_data[1: ], dtype=float)
    Y = data[:, -1]
    data_1 = data[np.where(Y == 1)]
    ss = StandardScaler() # 标准化数据，保证每个维度的特征数据方差为1，均值为0
    X = ss.fit_transform(data_1[:, :-1]) # 先拟合数据，再标准化，为保留原始分布特征
    return X

def knn_estimate(X_train, X_test, k, eps):
    Distance = np.abs(X_train - X_test)
    dist = sorted(Distance.tolist())[k]
    V = 2 * dist + eps
    N = len(X_train)
    prob = k / (N * V)
    return prob

if __name__ == "__main__":
    filename = "data/HWData3.csv"
    k = 5
    eps = 0.01

    X = load_data(filename)
    X_1 = X[:, 0]
    samples = np.arange(-3, 3, 0.05)
    prob = [knn_estimate(X_1, i, k, eps) for i in samples]   # 保存点的概率密度

    plt.title("K-nearest neighbor probability density estimation(k=5)")
    plt.plot(samples, prob, label="k=5")
    plt.legend(loc="best")
    plt.xlabel("Standardized x1")
    plt.ylabel("Probability")
    plt.show()