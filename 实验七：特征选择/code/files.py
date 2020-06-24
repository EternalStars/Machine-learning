import codecs
import pickle
import scipy.io as scio
import json 

# 读写.txt文件
def readtxt(filename):
    with codecs.open(filename, encoding="utf-8") as f:
        _data = f.readlines()
        data = [item.rstrip('\r\n') for item in _data]
    return data

def writetxt(filename, data):
    with codecs.open(filename, 'w', encoding='utf-8') as g:
        for line in data:
            g.write(str(line) + '\n')
    print("finish writing in %s !" % filename)

# 读写.pkl文件
def readpkl(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def writepkl(filename, data):
    g = open(filename, 'wb')
    pickle.dump(data, g)
    g.close()
    print("finish writing in %s !" % filename)

# 读写.mat文件
def readmat(filename):
    return scio.loadmat(filename)   # 返回结果为字典形式
    
def writemat(filename, data): # data必须为字典形式
    scio.savemat(filename, data)

# 写json文件，数据传入格式为字典，且字典中不能含有ndarray型数据，徐转成列表形式
def write_json(filename, data):
    with codecs.open(filename, 'w', encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4, separators=(',', ':')))
    print("finish write in", filename)
