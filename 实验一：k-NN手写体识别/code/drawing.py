from knn import knn_1
from knn import knn_2
from sklearn_knn import sklearn_knn as knn_s
import numpy as np

import matplotlib.pyplot as plt

def pic1(title, K):
    # 不同k值下分类器十折交叉验证结果数据(times=1,2,5)
    _k = np.array([i for i in range(1, K + 1)])

    knn_acc1 = knn_2(1, K)
    knn_acc2 = knn_2(2, K)
    knn_acc3 = knn_2(5, K)
    knn_acc4 = knn_2(10, K)

    plt.title(title[0])
    plt.plot(_k, knn_acc1, label='cycles=1')
    plt.plot(_k, knn_acc2, label='cycles=2')
    plt.plot(_k, knn_acc3, label='cycles=5')
    plt.plot(_k, knn_acc4, label='cycles=10')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()

def pic2(title, K):
    # 不同k值下两种分类器准确度结果比较数据
    _k = np.array([i for i in range(1, K + 1)])
    knn_train_acc, knn_test_acc = knn_1(K)
    sknn_train_acc, sknn_test_acc = knn_s(K)

    plt.title(title[1])
    plt.plot(_k, knn_train_acc, label='knn_train')
    plt.plot(_k, sknn_train_acc, label='sk_knn_train')
    plt.plot(_k, knn_test_acc, label='knn_test')
    plt.plot(_k, sknn_test_acc, label='sk_knn_test')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()

title = ['The result of 10-fold cross validation(cycles=1,2,5,10)', \
         'The result of k-NN and sklearn k-NN']
K = 10  # 近邻度K上限

pic1(title[0], K)
pic2(title[1], K)
