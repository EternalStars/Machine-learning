import csv
import numpy as np
from collections import Counter

def load_data(filename): 
    csv_reader = csv.reader(open(filename, encoding='utf-8'))
    data = []
    for line in csv_reader:
        item = line[0].strip().split(' ')
        data.append(item[:-10] + [item[-10:].index('1')]) # 拼接标签
    data = np.array(data, dtype=float)
    # Min_Max Normalization
    _data = data[:, :-1]
    _min = _data.min(1).reshape(-1, 1)
    _max = _data.max(1).reshape(-1, 1)
    _data = (_data - _min) / (_max - _min)
    data = np.concatenate((_data, data[:, -1].reshape(-1, 1)), axis=1) # 按列拼接，默认axis=0时按行拼接
    return data

# Euclidean Distance
def euclidean_dist(data_1, data_2):
    return np.linalg.norm(data_1 - data_2, ord=2, axis=1)

def knn_model(train, test, k):
    acc = 0
    for _test in test:
        ed = euclidean_dist(train[:, :-1], _test[:-1])
        ed_index = np.argsort(ed)[:k] # 返回升序排列的索引值，并取前k个
        k_labels = [train[item][-1] for item in ed_index] # 提取标签
        label = Counter(k_labels).most_common(1)[0][0] # 统计众数，两元素众数相同时返回排在前面的元素
        if label == _test[-1]:
            acc += 1
    rate = acc / len(test)
    return rate

# k折交叉验证
def k_fold_cross(times, k_fold, train, k):  # times:循环次数，k_fold:折数
    rate = []
    for i in range(0, times):       # 进行i次k折交叉验证
        np.random.shuffle(train)    # 随机打乱
        p = len(train) / k_fold     # 子集大小（向下取整）
        subset_num = [p for j in range(0, k_fold)] # 每个子集大小
        remainder = len(train) - int(p) * k_fold
        for j in range(0, remainder):
            subset_num[i] += 1
        subset = []
        count = 0
        for j in range(0, k_fold):  # 数据集切片
            subset.append(train[int(count):int(count+subset_num[j]), :])
            count += subset_num[j]
        for j in range(0, len(subset)): # k折交叉验证，对于k组数据，1组test set，k-1组train set
            _rate = 0
            for h in range(0, len(subset)):
                if (j + h) < len(subset):
                    _rate += knn_model(subset[j+h], subset[j], k)
                elif (j + h) >= len(subset):
                    _rate += knn_model(subset[j+h-len(subset)], subset[j], k) 
            rate.append(_rate / k_fold)
    rate = np.array(rate)
    res = np.sum(rate) / len(rate)
    return res

# 不通过交叉验证，直接采用grid search对测试集进行准确率分析
def knn_1(K):
    # 加载数据
    test = load_data('data/semeion_test.csv')
    train = load_data('data/semeion_train.csv')
    # 初始化参数
    k_max = K
    train_acc = np.empty(k_max)
    test_acc = np.empty(k_max)
    # grid search
    for k in range(0, k_max):
        train_acc[k] = knn_model(train, train, k+1)
        test_acc[k] = knn_model(train, test, k+1)
    print("train accuracy:", train_acc, "\ntest accuracy:", test_acc)
    return train_acc, test_acc

# 通过交叉验证的方式确定k值，再进行预测
def knn_2(cycles, K):
    # 加载数据
    test = load_data('data/semeion_test.csv')
    train = load_data('data/semeion_train.csv')
    
    # 参数赋值
    times = cycles   # k折交叉验证循环次数
    k_fold = 10 # k折交叉验证折数
    k_max = K  # knn中的k近邻数取值上限
    k_res = []   # 最后选取的合适的k值

    # 交叉验证选取k值
    rate_res = []
    for k in range(1, k_max + 1):
        rate = k_fold_cross(times, k_fold, train, k)
        print("当k为", k, "时，准确率为%.4f" %(100 * rate), "%")
        rate_res.append(rate)
        k_res.append(k)
    # 根据选择的k值，对测试集进行计算
    k_res_max = k_res[rate_res.index(max(rate_res))]
    knn_res = knn_model(train, test, k_res_max)
    print("\n训练集大小:", len(train), " 测试集大小:", len(test), "\nk =", k_res_max, \
        "\n准确率:%.4f" %(100*knn_res), "%")
    return np.array(rate_res) # k值1-10对应结果
