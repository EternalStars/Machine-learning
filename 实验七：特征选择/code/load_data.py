import files as f
import numpy as np

np.set_printoptions(threshold=np.inf)   # 完整输出
np.set_printoptions(suppress=True)      # 不以科学计数法输出

if __name__ == "__main__":
    data = f.readmat("data/urban.mat")
    data_X, data_Y = data['X'], data['Y']
    #print(data_X.shape)
    #print(data_Y.shape)
    data_Y = data_Y.reshape(-1)
    _len = {}
    for i in range(1, 10):
        _len[str(i)] = len(np.argwhere(data_Y == i))
    print(_len)
    #print(set(data_Y))
    '''
    data = np.concatenate((data_X, data_Y), axis=1)
    print(data.shape)
    f.writepkl("data/urban.pkl", data)
    f.writetxt("data/urban.txt", data)
    '''
