import files as f
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)   # 完整输出
np.set_printoptions(suppress=True)      # 不以科学计数法输出

# 各种分类器
class Classifier(object):
    def __init__(self, train, test):
        self.train_X = train[:, :-1]
        self.train_Y = train[:, -1]
        self.test_X = test[:, :-1]
        self.test_Y = test[:, -1]
        self.y_one_hot = label_binarize(self.test_Y, np.arange(9))
        #print(self.train_X.shape, self.train_Y.shape, self.test_X.shape, self.test_Y.shape)

    def _print(self, model, score, auc):
        return {'model': model, 'score': score, 'auc': auc}

    def knn(self):
        sknn = KNeighborsClassifier(n_neighbors=3)
        predict_prob_y = sknn.fit(self.train_X, self.train_Y).predict_proba(self.test_X) # 训练并返回分类结果概率
        score = sknn.score(self.test_X, self.test_Y)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_one_hot.ravel(), predict_prob_y.ravel())
        auc = metrics.auc(fpr, tpr)
        #auc = metrics.roc_auc_score(y_one_hot, predict_prob_y, average='macro')
        return self._print('KNN', score, auc)

    def NaiveBayes(self):
        snb = GaussianNB()
        predict_prob_y = snb.fit(self.train_X, self.train_Y).predict_proba(self.test_X)
        score = snb.score(self.test_X, self.test_Y)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_one_hot.ravel(), predict_prob_y.ravel())
        auc = metrics.auc(fpr, tpr)
        return self._print('NB', score, auc)
    
    def SVM(self):
        clf = svm.SVC(gamma='scale', probability=True)
        predict_prob_y = clf.fit(self.train_X, self.train_Y).predict_proba(self.test_X)
        score = clf.score(self.test_X, self.test_Y)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_one_hot.ravel(), predict_prob_y.ravel())
        auc = metrics.auc(fpr, tpr)
        return self._print('SVM', score, auc)

    def RandomForest(self):
        srf = RandomForestClassifier(n_estimators=25)
        predict_prob_y = srf.fit(self.train_X, self.train_Y).predict_proba(self.test_X)
        score = srf.score(self.test_X, self.test_Y)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_one_hot.ravel(), predict_prob_y.ravel())
        auc = metrics.auc(fpr, tpr)
        return self._print('RF', score, auc)

# Min-Max Normalize, [-1,1]
def normalize(data):
    return (data - np.mean(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Euclidean Dist
def euclidean_dist(data_1, data_2):
    return np.linalg.norm(data_1 - data_2, ord=2, axis=1)

# 两样本在特征A上的距离，max(A)-min(A)是基于全部样本还是基于sample2的？
def diff(feature, sample1, sample2, max_min_A):
    return np.abs(sample2[:, feature] - sample1[feature]) / max_min_A[feature]

# data->训练集 sampling_times->抽样次数 features_num->需求特征数 k->最近邻样本个数
def ReliefF(data, sampling_times, features_num, k):
    data_X, data_Y = normalize(data[:, :-1]), data[:, -1]
    data_classify = [data_X[np.argwhere(data_Y == i).reshape(-1)] for i in range(1, 10)]
    features_count = data_X.shape[1]  # 特征数
    weight = np.zeros(features_count)  # 初始化权重 
    max_min_A = np.max(data_X, axis=0) - np.min(data_X, axis=0) # 样本特征的最大值-最小值

    for i in range(sampling_times):
        # 从训练集中随机抽取样本R
        label = np.random.randint(1, 10, size=1)[0]     # 样本R的标签
        sampleR_index = np.random.randint(0, len(data_classify[label - 1]), size=1)[0]
        sampleR = data_classify[label - 1][sampleR_index]   # 随机抽取的样本R

        # 从同类及不同类样本中找到R的k个最近邻
        near = {}
        for j in range(1, 10):
            sample_j = data_classify[j - 1]
            dist = euclidean_dist(sampleR, sample_j)
            near[str(j)] = sample_j[np.argsort(dist)][:k]

        # 更新权重
        for A in range(features_count):
            near_hits_diff, near_misses_diff = 0, 0            
            for h in range(len(data_classify)):
                if h != label - 1:
                    near_misses_diff += np.sum(diff(A, sampleR, near[str(h + 1)], max_min_A)) / 9
                elif h == label - 1:
                    near_hits_diff += np.sum(diff(A, sampleR, near[str(h + 1)], max_min_A))
            weight[A] = weight[A] - (near_hits_diff - near_misses_diff) / (sampling_times * k)

    return np.argsort(-weight)[:features_num]

# data -> 训练集 || features_num -> 需求特征数
def MRMR(data, features_num):
    print("已运行特征量:", features_num)
    data_X, data_Y = normalize(data[:, :-1]), data[:, -1]
    fea_count = data_X.shape[1]    # 特征总量
    mrmr_fea = []  # 已选择特征集
    remain_fea = np.arange(fea_count)    # 剩余特征集

    # 计算初始最大互信息
    _info = np.array([metrics.mutual_info_score(data_X[:, i], data_Y) for i in range(fea_count)])
    _info_index = np.argsort(-_info)[0]
    mrmr_fea.append(_info_index)
    remain_fea = np.delete(remain_fea, _info_index)
    
    # 使用增量搜索实现mRMR
    for i in range(features_num - 1):
        res = []
        for j in range(len(remain_fea)):
            x_j = data_X[:, remain_fea[j]]
            info_remain = metrics.mutual_info_score(x_j, data_Y)
            info_mrmr = sum([metrics.mutual_info_score(x_j, data_X[:, t]) for t in mrmr_fea])
            mrmr_res = info_remain - (1 / (features_num - 1)) * info_mrmr  # 计算一轮迭代后的mrmr
            res.append(mrmr_res)
        _index = np.argmax(np.array(res))
        # 更新已选特征和剩余特征
        mrmr_fea.append(remain_fea[_index])
        remain_fea = np.delete(remain_fea, _index)
    
    return mrmr_fea

# 代码运行, fea_num->特征选择中需保留的特征数, noise->数据集添加的噪声数，默认None
def run(data, fea_num, method, noise=None):    # method=0 -> ReliefF || method=1 -> MRMR
    sampling_times = 50 # 抽样次数
    k_samples = 10  # 最近邻样本个数
    k_cross = 10    # k折交叉验证折数
    accuracy, auc = np.zeros((len(delta), 4)), np.zeros((len(delta), 4))

    for i in range(len(fea_num)):
        if method == 0:
            features = ReliefF(data, sampling_times, fea_num[i], k_samples)
        elif method == 1:
            features = MRMR(data, fea_num[i])
        _data = np.concatenate((data[:, features], data[:, -1].reshape(-1, 1)), axis=1)   # 特征选择后的样本

        # k折交叉验证运行代码(未分层处理)
        for j in range(k_cross):
            np.random.shuffle(_data)
            train, test = _data[int(len(_data) / k_cross):], _data[:int(len(_data) / k_cross)]
            clf = Classifier(train, test)
            _knn, _nb, _svm, _rf = clf.knn(), clf.NaiveBayes(), clf.SVM(), clf.RandomForest()
            accuracy[i][0] += _knn['score']
            accuracy[i][1] += _nb['score']
            accuracy[i][2] += _svm['score']
            accuracy[i][3] += _rf['score']
            auc[i][0] += _knn['auc']
            auc[i][1] += _nb['auc']
            auc[i][2] += _svm['auc']
            auc[i][3] += _rf['auc']
    
    accuracy, auc = accuracy / k_cross, auc / k_cross
    
    if noise == None:
        if method == 0:
            f.writepkl("result/acc_ReliefF.pkl", accuracy)
            f.writepkl("result/auc_ReliefF.pkl", auc)
        elif method == 1:
            f.writepkl("result/acc_MRMR.pkl", accuracy)
            f.writepkl("result/auc_MRMR.pkl", auc)
    elif noise != None:
        if method == 0:
            f.writepkl("result/acc_ReliefF_noise" + str(noise) + ".pkl", accuracy)
            f.writepkl("result/auc_ReliefF_noise" + str(noise) + ".pkl", auc)
        elif method == 1:
            f.writepkl("result/acc_MRMR_noise" + str(noise) + ".pkl", accuracy)
            f.writepkl("result/auc_MRMR_noise" + str(noise) + ".pkl", auc)

# 绘制acc图像
def draw_acc(fea_num, method, noise=None):
    plt.figure()
    _method = ['ReliefF', 'MRMR']
    lines = ['KNN', 'NaiveBayes', 'SVM', 'RandomForest']
    delta = np.array([1/6, 2/6, 3/6, 4/6, 5/6, 1])   # 特征权重阈值（选择排序前delta的特征）
    fea_count = data.shape[1] - 1   # 样本特征总量
    fea_num = np.array(fea_count * delta, dtype=int)    # 需求特征数
    if noise == None:
        accuracy = f.readpkl("result/acc_" + _method[method] + ".pkl")
        plt.title("The relationship between accuracy and different classifier(" + _method[method] + ")")
    elif noise != None:
        accuracy = f.readpkl("result/acc_" + _method[method] + "_noise" + str(noise) + ".pkl")
        plt.title("The relationship between accuracy and different classifier(" + _method[method] + ",noise=" + str(noise) + ")")
    for i in range(4):
        plt.plot(fea_num, accuracy[:, i], label=lines[i])
    plt.legend(loc="best")
    plt.xlabel("features count")
    plt.ylabel("accuracy")
    if noise == None:
        plt.savefig("pic/accuracy_" + _method[method] + ".png")
    elif noise != None:
        plt.savefig("pic/accuracy_" + _method[method] + "_noise" + str(noise) + ".png")
    #plt.show()

# 绘制auc图像
def draw_auc(fea_num, method, noise=None):
    plt.figure()
    _method = ['ReliefF', 'MRMR']
    lines = ['KNN', 'NaiveBayes', 'SVM', 'RandomForest']
    delta = np.array([1/6, 2/6, 3/6, 4/6, 5/6, 1])   # 特征权重阈值（选择排序前delta的特征）
    fea_count = data.shape[1] - 1   # 样本特征总量
    fea_num = np.array(fea_count * delta, dtype=int)    # 需求特征数
    if noise == None:
        auc = f.readpkl("result/auc_" + _method[method] + ".pkl")
        plt.title("The relationship between auc and different classifier(" + _method[method] + ")")
    elif noise != None:
        auc = f.readpkl("result/auc_" + _method[method] + "_noise" + str(noise) + ".pkl")
        plt.title("The relationship between auc and different classifier(" + _method[method] + ",noise=" + str(noise) + ")")
    for i in range(4):
        plt.plot(fea_num, auc[:, i], label=lines[i])
    plt.legend(loc="best")
    plt.xlabel("features count")
    plt.ylabel("auc")
    if noise == None:
        plt.savefig("pic/auc_" + _method[method] + ".png")
    elif noise != None:
        plt.savefig("pic/auc_" + _method[method] + "_noise" + str(noise) + ".png")

if __name__ == "__main__":
    # 先运行run，运行后注释，再逐步进行画图（为使绘制两个图象使用同一组数据且图象不重合）
    '''
    data = f.readpkl("data/urban.pkl")  # 训练集
    delta = np.array([1/6, 2/6, 3/6, 4/6, 5/6, 1])   # 特征权重阈值（选择排序前delta的特征）
    fea_count = data.shape[1] - 1   # 样本特征总量
    fea_num = np.array(fea_count * delta, dtype=int)    # 需求特征数
    '''
    
    # method=0 -> ReliefF || method=1 -> MRMR
    #run(data, fea_num, 1)
    #draw_acc(fea_num, 0)
    #draw_auc(fea_num, 0)

    noise_fea_num = [50, 100, 150, 200]
    num_list = []
    for i in noise_fea_num:
        print("样本特征添加：", i)
        data = f.readpkl("data/urban.pkl")  # 训练集
        noise = np.random.normal(loc=0, scale=5, size=(data.shape[0], i))
        data = np.concatenate((data[:, :-1], noise, data[:, -1].reshape(-1, 1)), axis=1)
        delta = np.array([1/6, 2/6, 3/6, 4/6, 5/6, 1])   # 特征权重阈值（选择排序前delta的特征）
        fea_count = data.shape[1] - 1   # 样本特征总量
        fea_num = np.array(fea_count * delta, dtype=int)    # 需求特征数
        
        #run(data, fea_num, 0, i)
        #run(data, fea_num, 1, i)
        #print("finish run!")
        draw_acc(fea_num, 0, i)
        draw_acc(fea_num, 1, i)
        draw_auc(fea_num, 0, i)
        draw_auc(fea_num, 1, i)