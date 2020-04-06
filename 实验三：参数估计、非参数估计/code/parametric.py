import csv
import math
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    with open(filename, encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        _data = [line for line in csv_reader]
    data = np.array(_data[1: ], dtype=float)
    ss = StandardScaler() # 标准化数据，保证每个维度的特征数据方差为1，均值为0
    X = ss.fit_transform(data[:, :-1]) # 先拟合数据，再标准化，为保留正态分布特征
    Y = data[:, -1]
    return X, Y

# 估计参数
def estimated_param(X):
    ave = np.average(X, axis=0) # 均值
    sub = X - ave
    cov = np.empty((X.shape[1], X.shape[1])) # 协方差
    for i in range(X.shape[1]):
        for j in range(i + 1):
            cov[i, j] = cov[j, i] = np.matmul(sub[:, i], sub[:, j]) / X.shape[0]
    return ave, cov

# 计算概率，这里传入的X和ave为一维向量
def calculate_prob(ave, cov, X):
    sub = (X - ave).reshape(X.shape[0], 1)
    coef_1 = 1 / math.pow(2 * math.pi, X.shape[0] / 2)
    coef_2 = 1 / math.sqrt(np.linalg.det(cov)) # det矩阵求逆
    coef_3 = math.exp(-1/2 * np.dot(sub.T, np.dot(np.linalg.inv(cov), sub))) # inv矩阵求行列式
    return coef_1 * coef_2 * coef_3

# 十折交叉验证计算结果
def cross_validate_param(X, Y):
    kf = KFold(n_splits=10, shuffle=False)
    acc_res = 0 # 计算平均正确率
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        params = []
        # 遍历所有标签，得到每一类标签下的参数值
        for i in range(1, 4):
            index = np.where(Y_train == i) # 返回所有断言成立时的索引值
            params.append(estimated_param(X_train[index]))
        acc = 0
        for i in range(len(X_test)):
            prob = [calculate_prob(ave, cov, X_test[i]) for ave, cov in params] # 计算三个参数下的概率
            y_hat = prob.index(max(prob)) + 1 # 概率最大值的标签作为结果
            if y_hat == Y_test[i]:
                acc += 1
        acc_res += acc / len(X_test)
    return acc_res / 10

if __name__ == "__main__":
    filename = "data/HWData3.csv"
    X, Y = load_data(filename)
    res = cross_validate_param(X, Y)
    print("多元高斯分布参数估计分类准确率为：%.2f" % (res * 100), '%')