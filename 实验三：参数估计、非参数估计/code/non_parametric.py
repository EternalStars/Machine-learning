import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors.kde import KernelDensity
from sklearn.preprocessing import StandardScaler

np.set_printoptions(suppress=True) # 不以科学计数法输出

def load_data(filename):
    with open(filename, encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        _data = [line for line in csv_reader]
    data = np.array(_data[1: ], dtype=float)
    ss = StandardScaler() # 标准化数据，保证每个维度的特征数据方差为1，均值为0
    X = ss.fit_transform(data[:, :-1]) # 先拟合数据，再标准化，为保留正态分布特征
    Y = data[:, -1]
    return X, Y

def cross_validate_nonparam(X, Y, kernel, bandwidth):
    kf = KFold(n_splits=10, shuffle=False)
    acc_res = 0 # 计算平均正确率
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        _kd = []
        # 遍历所有标签
        for i in range(1, 4):
            index = np.where(Y_train == i) # 返回所有断言成立时的索引值
            _kd.append(KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X_train[index]))
        acc = 0
        prob = np.array([item.score_samples(X_test) for item in _kd]).T
        for i in range(len(X_test)):
            y_hat = prob[i].tolist().index(max(prob[i])) + 1 # 概率最大值的标签作为结果
            if y_hat == Y_test[i]:
                acc += 1
        acc_res += acc / len(X_test)
    return acc_res / 10

def fit_bandwidth(X, Y, kernel):
    acc = []
    _bandwidth = np.arange(0.01, 2.01, 0.01)
    for bw in _bandwidth:
        acc.append(cross_validate_nonparam(X, Y, kernel, bw))
    acc = np.array(acc)
    bands_maxacc = _bandwidth[np.where(acc == max(acc))] # 准确度最大时的bandwith
    return _bandwidth, acc, bands_maxacc

def draw(x, y, plot_name):
    plt.title("The accuracy comparison of 5 kernels in different bandwidth")
    for i in range(len(y)):
        plt.plot(x, y[i], label=plot_name[i])
    plt.legend(loc="best")
    plt.xlabel("bandwidth")
    plt.ylabel("accuracy")
    plt.show()

if __name__ == "__main__":
    filename = "data/HWData3.csv"
    kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear']
    X, Y = load_data(filename)
    _acc = [] # 存储所有核函数在不同bandwidth下的准确率，以便作图

    for kernel in kernels:
        _bandwidth, acc, bands_maxacc = fit_bandwidth(X, Y, kernel)
        print('kernel=%s,' % kernel, '最大正确率%.4f' % (max(acc) * 100), '%' ,"\n正确率最大时的bandwidth:", bands_maxacc)
        _acc.append(acc)
    
    #draw(_bandwidth, _acc, kernels)