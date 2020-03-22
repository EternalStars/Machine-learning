import numpy as np
import pickle
import matplotlib.pyplot as plt
import math
import time

from regression import init_data
from regression import read_pkl
from regression import model
from regression import batch_gradient_descent
from regression import L2_BGD

# 使用留一法的RMSE
def RMSE_LeaveOne():
    data = read_pkl("data/winequality-white.pkl")
    X, Y, theta = init_data(data)
    iters_max = 20
    learning_rate = 0.005
    
    _iters = np.array([i for i in range(5, iters_max, 10)])
    _rmse = []  # 均方误差保存的列表

    for iters in _iters:
        _err = 0
        # 使用留一法
        for i in range(0, len(data)):
            if i % 500 == 0:
                print("now in iters: ", iters, " data: ", i)
            test_X = X[i, :]
            test_Y = Y[i, :]
            train_X = np.delete(X, i, axis=0)
            train_Y = np.delete(Y, i, axis=0)
            res_theta, res_cost = model(train_X, train_Y, theta, "BGD", learning_rate, iters)
            _err += (test_Y[0] - np.dot(test_X, res_theta)[0]) ** 2
        _rmse.append(math.sqrt(_err / len(data)))
    
    plt.title("The relationship between RMSE and iterations with BGD")
    plt.plot(_iters, _rmse, label="BGD")
    plt.legend(loc="best")
    plt.xlabel("iterations")
    plt.ylabel("RMSE")
    plt.show()

# 使用自己划分的训练集和测试集计算出的RMSE
def RMSE():
    train = read_pkl("data/train.pkl")
    test = read_pkl("data/test.pkl")
    train_X, train_Y, theta = init_data(train)
    test_X, test_Y, test_theta = init_data(test) # test_theta可以忽略，仅仅为了对应函数返回值

    iters_max = 1000
    learning_rate = 0.005
    _iters = np.array([i for i in range(0, iters_max, 10)])
    _rmse = []  # 均方误差保存的列表
    _rmse_2 = []

    # 计算RMSE
    for iters in _iters:
        res_theta, res_cost = model(train_X, train_Y, theta, "BGD", learning_rate, iters)
        res_theta_2, res_cost_2 = model(train_X, train_Y, theta, "SGD", learning_rate, iters)
        err = np.sum((test_Y - np.dot(test_X, res_theta)) ** 2) / len(test_X)
        err_2 = np.sum((test_Y - np.dot(test_X, res_theta_2)) ** 2) / len(test_X)
        _rmse.append(math.sqrt(err))
        _rmse_2.append(math.sqrt(err_2))
    
    plt.title("The relationship between RMSE and iterations with BGD and SGD(500-1000)")
    plt.plot(_iters, _rmse, label="BGD")
    plt.plot(_iters, _rmse_2, label="SGD")
    plt.legend(loc="best")
    plt.xlabel("iterations")
    plt.ylabel("RMSE")
    plt.show()

# 对比学习率
def contrast_learning_rate():
    data = read_pkl("data/winequality-white.pkl")
    X, Y, theta = init_data(data)
    iters = 1000
    gradient = "BGD"
    learning_rate = [0.001, 0.002, 0.003, 0.006, 0.01, 0.03, 0.1]
    
    plt.title("The relationship between cost and iterations in different learning rates")
    for rate in learning_rate:
        _theta, cost = model(X, Y, theta, gradient, rate, iters)
        label_ = "rate=" + str(rate)
        plt.plot(np.arange(iters), cost, label=label_)
    plt.legend(loc="best")
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.show()

# 线性分类器，X：要预测的数据，theta：拟合的参数
def classify(theta, X):
    y_hat = np.dot(X, theta).reshape(1, -1)[0] # 预测值
    for i in range(0, len(y_hat)):
        if y_hat[i] >= 9:
            y_hat[i] = 9
            continue
        if y_hat[i] <= 3:
            y_hat[i] = 3
            continue
        decimal = y_hat[i] - math.floor(y_hat[i])
        if decimal >= 0.5: # 向上取整
            y_hat[i] = math.ceil(y_hat[i])
        elif decimal < 0.5: # 向下取整
            y_hat[i] = math.floor(y_hat[i])
    return y_hat 

# 比较BGD和SGD分类精度
def contrast_BGD_SGD():
    train = read_pkl("data/train.pkl")
    test = read_pkl("data/test.pkl")
    train_X, train_Y, theta = init_data(train)
    test_X, test_Y, test_theta = init_data(test)

    iters_max = 1000
    learning_rate = 0.05
    _iters = np.array([i for i in range(500, iters_max, 10)])
    
    _bgd_acc = []
    _sgd_acc = []
    for iters in _iters:
        # 分类精度对比
        bgd_theta, bgd_cost = model(train_X, train_Y, theta, "BGD", learning_rate, iters)
        sgd_theta, sgd_cost = model(train_X, train_Y, theta, "SGD", learning_rate, iters)
        bgd_yhat = classify(bgd_theta, test_X)
        sgd_yhat = classify(sgd_theta, test_X)
    
        bgd_acc = 0
        sgd_acc = 0
        test_Y = test_Y.reshape(1, -1)[0]
        for i in range(0, len(test_Y)):
            if bgd_yhat[i] == test_Y[i]:
                bgd_acc += 1
            if sgd_yhat[i] == test_Y[i]:
                sgd_acc += 1
        _bgd_acc.append(bgd_acc)
        _sgd_acc.append(sgd_acc)
        
        #print("BGD Accuracy: ", 100 * (bgd_acc / len(bgd_yhat)), "%")
        #print("SGD Accuracy: ", 100 * (sgd_acc / len(sgd_yhat)), "%")

    plt.title("The accuracy comparison of BGD and SGD")
    plt.plot(_iters, _bgd_acc, label="BGD")
    plt.plot(_iters, _sgd_acc, label="SGD")
    plt.legend(loc="best")
    plt.xlabel("iterations")
    plt.ylabel("accuracy(%)")
    plt.show()

# BGD和SGD执行时间比较
def contrast_BGD_SGD_time():
    train = read_pkl("data/train.pkl")
    test = read_pkl("data/test.pkl")
    train_X, train_Y, theta = init_data(train)
    test_X, test_Y, test_theta = init_data(test)

    iters_max = 1000
    learning_rate = 0.05
    _iters = np.array([i for i in range(0, iters_max, 10)])
    
    _bgd_time = []
    _sgd_time = []
    for iters in _iters:
        # 执行时间对比
        tic_bgd = time.time()
        bgd_theta, bgd_cost = model(train_X, train_Y, theta, "BGD", learning_rate, iters)
        toc_bgd = time.time()
        bgd_time = 1000 * (toc_bgd - tic_bgd)

        tic_sgd = time.time()
        sgd_theta, sgd_cost = model(train_X, train_Y, theta, "SGD", learning_rate, iters)
        toc_sgd = time.time()
        sgd_time = 1000 * (toc_sgd - tic_sgd)
        
        _bgd_time.append(bgd_time)
        _sgd_time.append(sgd_time)    
        #print("BGD Execution Time: ", 1000 * (toc_bgd - tic_bgd), "ms")
        #print("SGD Execution Time: ", 1000 * (toc_sgd - tic_sgd), "ms")
    
    plt.title("The execution time comparison of BGD and SGD")
    plt.plot(_iters, _bgd_time, label="BGD")
    plt.plot(_iters, _sgd_time, label="SGD")
    plt.legend(loc="best")
    plt.xlabel("iterations")
    plt.ylabel("execution time(ms)")
    plt.show()

# 比较加了L2正则化后的BGD和普通BGD的结果
def contrast_BGD_L2BGD():
    train = read_pkl("data/train.pkl")
    test = read_pkl("data/test.pkl")
    train_X, train_Y, theta = init_data(train)
    test_X, test_Y, test_theta = init_data(test)

    iters_max = 1000
    learning_rate = 0.05
    _lambda = 100
    _iters = np.array([i for i in range(500, iters_max, 10)])

    _bgd_acc = []
    _L2bgd_acc = []

    for iters in _iters:
        bgd_theta, bgd_cost = batch_gradient_descent(train_X, train_Y, theta, learning_rate, iters)
        L2bgd_theta, L2bgd_cost = L2_BGD(train_X, train_Y, theta, _lambda, learning_rate, iters)

        # 分类精度
        bgd_yhat = classify(bgd_theta, test_X)
        L2bgd_yhat = classify(L2bgd_theta, test_X)
    
        bgd_acc = 0
        L2bgd_acc = 0
        test_Y = test_Y.reshape(1, -1)[0]
        for i in range(0, len(test_Y)):
            if bgd_yhat[i] == test_Y[i]:
                bgd_acc += 1
            if L2bgd_yhat[i] == test_Y[i]:
                L2bgd_acc += 1
        _bgd_acc.append(100 * (bgd_acc / len(bgd_yhat)))
        _L2bgd_acc.append(100 * (L2bgd_acc / len(L2bgd_yhat)))
        #print("BGD Accuracy: ", 100 * (bgd_acc / len(bgd_yhat)), "%")
        #print("L2-BGD Accuracy: ", 100 * (L2bgd_acc / len(L2bgd_yhat)), "%")

    plt.title("The accuracy comparison of BGD and L2-BGD")
    plt.plot(_iters, _bgd_acc, label="BGD")
    plt.plot(_iters, _L2bgd_acc, label="L2-BGD")
    plt.legend(loc="best")
    plt.xlabel("iterations")
    plt.ylabel("accuracy(%)")
    plt.show()

if __name__ == "__main__":
    #RMSE()
    #contrast_learning_rate()
    #contrast_BGD_SGD()
    #contrast_BGD_SGD_time()
    #contrast_BGD_L2BGD()
    pass