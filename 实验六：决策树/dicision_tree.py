import numpy as np
import codecs
from sklearn.metrics import accuracy_score

feature_dict = {"色泽": ["青绿", "乌黑", "浅白"],
                "根蒂": ["蜷缩", "稍蜷", "硬挺"],
                "敲声": ["浊响", "沉闷", "清脆"],
                "纹理": ["清晰", "稍糊", "模糊"]
                }
lable_list = ["否", "是"]
feature_list = ["色泽", "根蒂", "敲声", "纹理"]

def load_txt(path):
    ans = []
    with codecs.open(path, "r", "GBK") as f:
        line = f.readline()
        line = f.readline()
        while line:
            d = line.rstrip("\r\n").split(',')
            #print(d)
            re = []
            # 输入编号方便追踪
            re.append(int(d[0]))
            re.append(feature_dict.get("色泽").index(d[1]))
            re.append(feature_dict.get("根蒂").index(d[2]))
            re.append(feature_dict.get("敲声").index(d[3]))
            re.append(feature_dict.get("纹理").index(d[4]))

            re.append(lable_list.index(d[-1]))
            ans.append(np.array(re))
            line = f.readline()
    return np.array(ans)

class Node:
    def __init__(self, attr, label, v):
        # label == pi是非叶节点
        # attr == pi 是叶节点
        self.attr = attr
        self.label = label
        self.attr_v = v
        self.children = []

def is_same_on_attr(X, attrs):#验证属性上的值是否均相同
    X_a = X[:, attrs]
    target = X_a[0]
    for r in range(X_a.shape[0]):
        row = X_a[r]
        if (row != target).any():
            return False
    return True

def ent(D):
    # D is a 1d np array which actually is Y
    s = 0
    for k in set(D):
        p_k = np.sum(np.where(D == k, 1, 0)) / np.shape(D)[0]
        if p_k == 0:
            # 此时Pklog2Pk 定义为 0
            continue
        s += p_k * np.log2(p_k)
    return -s

def gain(X, Y, attr):
    # X, Y 是numpy arrary attr是某个特征的index
    x_attr_col = X[:, attr]
    ent_Dv = []
    weight_Dv = []
    # 离散值处理
    for x_v in set(x_attr_col):
        index_x_equal_v = np.where(x_attr_col == x_v)
        y_x_equal_v = Y[index_x_equal_v]
        ent_Dv.append(ent(y_x_equal_v))
        weight_Dv.append(np.shape(y_x_equal_v)[0] / np.shape(Y)[0])
    return ent(Y) - np.sum(np.array(ent_Dv) * np.array(weight_Dv))

def dicision_tree_init(X, Y, attrs, root, purity_cal):
    # 递归基
    if len(set(Y)) == 1:
        root.attr = np.pi
        root.label = Y[0]
        return None

    if len(attrs) == 0 or is_same_on_attr(X, attrs):
        root.attr = np.pi
        # Y 中出现次数最多的label设定为node的label
        root.label = np.argmax(np.bincount(Y))
        return None

    # 计算每个attr的划分收益
    purity_attrs = []
    for i, a in enumerate(attrs):
        p = purity_cal(X, Y, a)
        purity_attrs.append(p)
    #print(purity_attrs)
    chosen_index = purity_attrs.index(max(purity_attrs))
    chosen_attr = attrs[chosen_index]

    root.attr = chosen_attr
    root.label = np.pi

    del attrs[chosen_index]

    x_attr_col = X[:, chosen_attr]
    # 离散数据处理
    for x_v in set(X[:, chosen_attr]):
        n = Node(-1, -1, x_v)
        root.children.append(n)
        # 不可能Dv empty 要是empty压根不会在set里
        # 选出 X[attr] == x_v的行

        index_x_equal_v = np.where(x_attr_col == x_v)
        X_x_equal_v = X[index_x_equal_v]
        Y_x_equal_v = Y[index_x_equal_v]
        dicision_tree_init(X_x_equal_v, Y_x_equal_v, attrs, n, purity_cal)

def dicision_tree_predict(x, tree_root):
    if tree_root.label != np.pi:
        return tree_root.label

    # 决策
    if tree_root.label == np.pi and tree_root.attr == np.pi:
        print("err!")
        return None

    chose_attr = tree_root.attr
    # 寻找自己应该进入哪个分支
    for child in tree_root.children:
        if child.attr_v == x[chose_attr]:
            return dicision_tree_predict(x, child)
    return None

if __name__ == '__main__':
    ans = load_txt("Watermelon-train1.csv")
    X_train = ans[:, 1: -1]
    Y_train = ans[:, -1]
    Y_train.astype(np.int64)
    # print(X_train)
    # print(Y_train)

    test_data = load_txt("Watermelon-test1.csv")
    X_test = test_data[:, 1:-1]
    Y_test = test_data[:, -1]

    r = Node(-1, -1, -1)
    attrs = [0, 1, 2, 3]

    dicision_tree_init(X_train ,Y_train, attrs, r,gain)

    y_predict = []
    for i in range(X_test.shape[0]):
        x = X_test[i]
        y_p = dicision_tree_predict(x, r)
        y_predict.append(y_p)

    acc = accuracy_score(Y_test, y_predict)
    print('accuracy:',acc)