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
    #np.random.shuffle(data)    # 随机打乱
    ss = StandardScaler()
    X = ss.fit_transform(data[:, :-1])
    Y = data[:, -1]
    return X, Y

# 计算欧氏距离
def euclidean_dist(data_1, data_2):
    return np.linalg.norm(data_1 - data_2, ord=2, axis=1)

# 最近邻决策规则
def nn_Decision(X_train, Y_train, X_test):
    y_hat = []
    for item in X_test:
        ed = euclidean_dist(X_train, item)
        y_hat.append(Y_train[ed.tolist().index(min(ed))])
    return np.array(y_hat, dtype=int)

# 十折交叉验证计算结果
def cross_validate_nn(X, Y):
    kf = KFold(n_splits=10, shuffle=False)
    acc_res = 0 # 计算平均正确率
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        y_hat = nn_Decision(X_train, Y_train, X_test)
        acc = 0
        for i in range(len(X_test)):
            if y_hat[i] == Y_test[i]:
                acc += 1
        acc_res += acc / len(X_test)
    return acc_res / 10

if __name__ == "__main__":
    filename = "data/HWData3.csv"
    X, Y = load_data(filename)
    res = cross_validate_nn(X, Y)
    print("最近邻决策分类准确率为：%.2f" % (res * 100), '%')