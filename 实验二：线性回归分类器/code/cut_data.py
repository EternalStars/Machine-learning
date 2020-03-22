import csv
import numpy as np
import pickle

def load_data(filename):
    with open(filename, encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        _data = [line for line in csv_reader]
    data = np.array(_data[1: ], dtype=float)
    #np.set_printoptions(suppress=True) # 不以科学计数法输出
    return data

# 将序列化对象存储在pickle文件中
def write_pkl(filename, data):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()
    print("finish writing in %s !" % filename)

# 划分训练集和测试集
def cut_data(filename):
    data = load_data(filename);
    np.random.shuffle(data)
    train_count = int(len(data) / 10 * 7)
    write_pkl("data/train.pkl", data[: train_count])
    write_pkl("data/test.pkl", data[train_count: ])

#data = load_data("data/winequality-white.csv")
#write_pkl("data/winequality-white.pkl", data)
#cut_data("data/winequality-white.csv")