import numpy as np
import pickle
import matplotlib.pyplot as plt
import math
from cut_data import write_pkl
import time

# 读取pickle文件中的序列化对象
def read_pkl(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

# 使归一化后的结果在[-1,1]区间内，可以加快收敛
def normalize(data):
    # axis=0，压缩行，对各列求均值，返回1*n矩阵
    return (data - np.mean(data, 0)) / (np.max(data, 0) - np.min(data, 0))

# 损失函数，X:样本特征矩阵，Y:样本的真实值，theta:参数矩阵
def cost_function(X, Y, theta): 
    inner = np.power((np.dot(X, theta) - Y), 2)
    return np.sum(inner) / (2 * len(inner))

# 批量梯度下降，learning_rata:学习率 iters:迭代次数
def batch_gradient_descent(X, Y, theta, learning_rate, iters):
    parameters = X.shape[1]
    temp = np.zeros((parameters, 1))    # 暂时存储梯度下降后的参数值，以待同步更新
    cost = np.zeros(iters)  # 记录损失值
    for i in range(0, iters):
        error = np.dot(X, theta) - Y
        for j in range(0, parameters):
            X_j = X[:, j].reshape(-1, 1)
            temp[j] = theta[j] - learning_rate / X.shape[0] * np.sum(error * X_j)
        theta = temp
        cost[i] = cost_function(X, Y, theta)
    return theta, cost

# 随机梯度下降
def stochastic_gradient_descent(X, Y, theta, learning_rate, iters):
    parameters = X.shape[1]
    temp = np.zeros((parameters, 1))    # 暂时存储梯度下降后的参数值，以待同步更新
    cost = np.zeros(iters)  # 记录损失值
    for i in range(0, iters):
        start = np.random.randint(0, X.shape[0], 1)[0]
        error = np.dot(X[start], theta)[0] - Y[start][0]
        for j in range(0, parameters):
            temp[j] = theta[j] - learning_rate * error * X[start][j]
        theta = temp
        cost[i] = cost_function(X, Y, theta)
    return theta, cost

# L2正则化后的批量梯度下降
def L2_BGD(X, Y, theta, _lambda, learning_rate, iters):
    parameters = X.shape[1]
    temp = np.zeros((parameters, 1))    # 暂时存储梯度下降后的参数值，以待同步更新
    cost = np.zeros(iters)  # 记录损失值
    regularization = 1 - learning_rate * _lambda / X.shape[0]
    for i in range(0, iters):
        error = np.dot(X, theta) - Y
        X_0 = X[:, 0].reshape(-1, 1)
        temp[0] = theta[0] - learning_rate / X.shape[0] * np.sum(error * X_0)
        for j in range(1, parameters):
            X_j = X[:, j].reshape(-1, 1)
            temp[j] = theta[j] * regularization - learning_rate / X.shape[0] * np.sum(error * X_j)
        theta = temp
        cost[i] = cost_function(X, Y, theta)
    return theta, cost

# 初始化数据集
def init_data(data):
    X = normalize(data[:, :-1])
    X_0 = np.ones(X.shape[0])
    # 初始化参数
    X = np.insert(X, 0, X_0, axis=1)
    Y = data[:, -1].reshape(-1, 1)
    theta = np.zeros((X.shape[1], 1))
    return X, Y, theta

# 模型训练
def model(X, Y, theta, gradient, learning_rate, iters):
    if(gradient == "BGD"):
        theta_gd, cost = batch_gradient_descent(X, Y, theta, learning_rate, iters)
    elif(gradient == "SGD"):
        theta_gd, cost = stochastic_gradient_descent(X, Y, theta, learning_rate, iters)
    return theta_gd, cost



#np.set_printoptions(suppress=True)
#print("随机：", theta_sgd.reshape(1,-1))
#print("批量：", theta_bgd.reshape(1,-1))