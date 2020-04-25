import math
import codecs
import numpy as np
import prettytable as pt
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)   # 完整输出
np.set_printoptions(suppress=True)      # 不以科学计数法输出

# 读取文件，每行以矩阵形式输出
def readfile(filename):
    with codecs.open(filename, encoding='utf-8') as f:
        _list = f.readlines()
    for i in range(0, len(_list)):
        _list[i] = _list[i].rstrip('\n')
        _list[i] = _list[i].rstrip('\r')
    data = np.array([item.split(',') for item in _list], dtype=float)
    return data

# prior_pr,样本为类别c的先验概率，格式为一维数组，train和test为[array([[]])]
def NaiveBayes_Classifier(train, test, prior_pr):
    ave = np.array([np.mean(train[i], axis=0) for i in range(3)])    # 均值
    std = np.array([np.std(train[i], axis=0) for i in range(3)])     # 标准差
    acc = 0 # 记录正确个数
    confusion_matrix = np.zeros((3, 3)) # 混淆矩阵
    label_score = [] # 真实标签和对应得分的列表

    for i in range(3):
        for t in test[i]:
            score = []
            for j in range(3):
                comp_1 = np.log((2 * math.pi) ** 0.5 * std[j])  # 取自然对数减小计算成本
                comp_2 = np.power(t - ave[j], 2) / (2 * np.power(std[j], 2))
                guassian_score = -1 * np.sum(comp_1 + comp_2) + math.log(prior_pr[j])
                score.append(guassian_score)
            label_score.append([i + 1, score[0], score[1], score[2]])
            y_hat = score.index(max(score))     # 取最大值为预测值
            if y_hat == i:
                acc += 1
                confusion_matrix[i][i] += 1
            else:
                confusion_matrix[i][y_hat] += 1
    return acc, confusion_matrix, label_score

# 计算通用性能指标，传入参数为混淆矩阵
def performance_metrics(cm): # cm: confusion_matrix
    accuracy = np.trace(np.mat(cm)) / np.sum(cm) # np.trace 矩阵的迹
    recall = np.array([cm[i][i] / np.sum(cm[i]) for i in range(3)]) # 召回率
    mean_recall = np.mean(recall) # 平均召回率
    F_measure = 2 / (1 / accuracy + 1 / mean_recall) # F1值

    print("Confusion Matrix:")
    tb = pt.PrettyTable()
    tb.field_names = ["real\predit", "predict_1", "predict_2", "predict_3"]
    tb.add_row(["real_1"] + cm[0].tolist())
    tb.add_row(["real_2"] + cm[1].tolist())
    tb.add_row(["real_3"] + cm[2].tolist())
    print(tb)

    print("Accuracy: %.4f" % (accuracy * 100), "%")
    print("Recall(label=1): %.4f" % (recall[0] * 100), "%")
    print("Recall(label=2): %.4f" % (recall[1] * 100), "%")
    print("Recall(label=3): %.4f" % (recall[2] * 100), "%")
    print("Average Recall: %.4f" % (mean_recall * 100), "%")
    print("F_measure:", F_measure)

# Min-Max Normalization => (0, 1)
def mm_Normalization(data):
    Y = data[:, 0].reshape(-1, 1)
    X = data[:, 1:]
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    return np.hstack((Y, X))

# 绘制roc曲线并计算auc
def roc_auc(label_score, i):    # i: 标签为i+1的数据对应的roc曲线
    label_score = mm_Normalization(label_score)
    N = label_score.shape[0]    # 数据集总量
    index = np.argsort(label_score[:, i+1]) # 升序排列
    i_score = label_score[index]
    for j in range(N):
        if i_score[j][0] != i + 1:
            i_score[j][0] = 0
        else:
            i_score[j][0] = 1
    Y_real = i_score[:, 0]  # 真实标签
    score = i_score[:, i+1] # 得分
    TPR = []    # 纵坐标
    FPR = []    # 横坐标
    for s in range(N):  # 遍历间断点
        Y_predict = np.ones(N)
        for k in range(s):
            Y_predict[k] = 0  # 得分小于间断点为负例，得分大于等于间断点为正例
        TP, FP, FN, TN = 0, 0, 0, 0
        for t in range(N):  # 遍历标签
            if Y_real[t] == 1 and Y_predict[t] == 1:
                TP += 1
            if Y_real[t] == 0 and Y_predict[t] == 1:
                FP += 1
            if Y_real[t] == 1 and Y_predict[t] == 0:
                FN += 1
            if Y_real[t] == 0 and Y_predict[t] == 0:
                TN += 1
        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))
    # 计算AUC
    pos_rank = 0    # 正例rank加和
    for a in range(N):   # 求AUC，即ROC曲线下的面积
        if Y_real[a] == 1:
            pos_rank += a + 1
    M = len(np.argwhere(Y_real == 1))   # 正例个数
    N = len(np.argwhere(Y_real == 0))   # 负例个数
    AUC = (pos_rank - M * (M + 1) / 2) / (M * N)

    plt.title("ROC(label=" + str(i + 1) + ")   AUC: %.4f" % AUC)
    plt.plot(FPR, TPR, label=("label" + str(i + 1)))
    plt.legend(loc="best")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

if __name__ == "__main__":
    data = readfile("data/wine.data")
    Y = np.array(data[:, 0], dtype=int)  # (178,)
    X = data[:, 1:] # (178, 13)
    
    labels = sorted(list(set(Y)))   # 排序后的标签
    labels_num = np.array([len(np.argwhere(Y == c)) for c in labels])   # 每一类数据的数量
    prior_pr = labels_num / len(Y)  # prior_pr，样本为类c的先验概率
    # 由于每一类下维度不统一，用列表存储X_classify
    X_classify = [X[np.argwhere(Y == c).reshape(1, -1)[0]] for c in labels] # 分类后的data
    test_len = [int(round(i / 10)) for i in labels_num]  # 每一层测试集数据大小
    
    # 分层十折交叉验证
    acc = 0 # 分类正确数
    cm = np.zeros((3, 3))   # 混淆矩阵
    label_score = []    # 真实标签和对应得分的列表
    for i in range(10):
        train = []  # 格式为[array([[]])]，下同
        test = []
        for j in range(3):
            if (i + 1) * test_len[j] > labels_num[j] or i == 9: # i==9是因为标签为2的数据有71个
                test.append(X_classify[j][i*test_len[j]:, :])
                train.append(X_classify[j][:i*test_len[j], :])
            else:
                test.append(X_classify[j][i*test_len[j]:(i+1)*test_len[j], :])
                train.append(np.vstack((X_classify[j][:i*test_len[j], :], X_classify[j][(i+1)*test_len[j]:, :])))
        _acc, _cm, _ls = NaiveBayes_Classifier(train, test, prior_pr)
        acc += _acc
        cm += _cm
        label_score += _ls
    label_score = np.array(label_score)
    label_score = label_score[np.argsort(label_score[:, 0])]

    print("The total capacity of data set is:", Y.shape[0], "\nThe correct number of predictions is:", acc)
    print("The accuracy of Naive-Bayes classifier is: %.2f" % (acc * 100 / Y.shape[0]), "%")
    # 计算常用指标
    performance_metrics(cm)
    # ROC曲线及AUC值计算
    #roc_auc(label_score, 0)
    #roc_auc(label_score, 1)
    roc_auc(label_score, 2)