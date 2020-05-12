import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

def generate_samples(count, centers, std):
    X, Y = make_blobs(n_samples=count, centers=centers, cluster_std=std)
    np.savetxt(os.path.dirname(__file__) + '/data/data_x.dat', X)
    np.savetxt(os.path.dirname(__file__) + '/data/data_y.dat', Y)
    print("Finished generating samples in /data/data_x.dat and /data/data_y.dat")
    return X, np.array(Y, dtype=int)

# ------Hausdorff Distance methods------
# Hausdorff Distance
def dist_h(x, z):
    return np.linalg.norm(np.max(x) - np.min(z))
def dist_H(x, z):
    return max(dist_h(x, z), dist_h(z, x))

# single linkage: minimum distance
def dist_min(c_i, c_j):
    return min([dist_H(x, z) for x in c_i for z in c_j])
# complete linkage: maximum distance
def dist_max(c_i, c_j):
    return max([dist_H(x, z) for x in c_i for z in c_j])
# average linkage: average distance
def dist_ave(c_i, c_j):
    d_sum = sum([dist_H(x, z) for x in c_i for z in c_j])
    return d_sum / (len(c_i) * len(c_j))

# ------Euclidean Distance methods------
# Euclidean distance
def dist_emin(c_i, c_j):
    return min([np.sum((x - y) ** 2) for x in c_i for y in c_j])
def dist_emax(c_i, c_j):
    return max([np.sum((x - y) ** 2) for x in c_i for y in c_j])
def dist_eave(c_i, c_j):
    d_sum = sum([np.sum((x - y) ** 2) for x in c_i for y in c_j])
    return d_sum / (len(c_i) * len(c_j))

# 自底向上聚合策略
def AGNES(X, Y, dist, k): # dist:距离度量函数 k:聚类簇数 
    N = X.shape[0]
    CIndex = [[i] for i in range(N)] # 点的索引标号
    C = [[x] for x in X] # 初始化单样本聚类簇 C: [[array()]]

    # 初始化聚类簇距离矩阵
    MAX_DIS = 1e3
    M = np.zeros((N, N)) + MAX_DIS
    for i in range(N):
        for j in range(i + 1, N):
            M[i][j] = M[j][i] = dist(C[i], C[j])

    cluster_count = X.shape[0] # 当前聚类簇个数
    while cluster_count > k:
        _index = np.argmin(M)
        ci_index, cj_index = int(_index / cluster_count), _index % cluster_count
        C[ci_index] += C[cj_index]
        CIndex[ci_index] += CIndex[cj_index]
        del C[cj_index]
        del CIndex[cj_index]
        M = np.delete(M, cj_index, axis=0)
        M = np.delete(M, cj_index, axis=1)
        if ci_index > cj_index:
            ci_index -= 1
        for j in range(M.shape[0]):
            if j != ci_index:
                M[ci_index][j] = M[j][ci_index] = dist(C[ci_index], C[j])
        cluster_count -= 1
        if (N - cluster_count) % 100 == 0:
            print("Number of remaining clusters %d" % (N - cluster_count))
    return C, CIndex

def creat_plot(X, Y, Y_hat, k, name):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    colors = 'rgbyckm' # 每个簇的样本标记不同的颜色
    markers = 'o^sP*DX'
    for i in range(len(Y)):
        ax.scatter(X[i,0], X[i,1], label="cluster %d" % Y[i], \
		color=colors[y_hat[i] % len(colors)], marker=markers[Y[i] % len(markers)], alpha=0.5)

    #ax.legend(loc="best",framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.set_title(name + '(k = %d' % k + ')')
    plt.savefig(os.path.dirname(__file__) + '/pic/' + name + '(k = %d' % k + ')')

if __name__ == "__main__":
    count = 2000
    centers = [[1,1,1],[1,3,3],[3,6,5],[2,6,8]] # 产生聚类的中心
    std = 0.5
    k = 5
    DISTANCE_E = [dist_emin, dist_emax, dist_eave]  # 欧氏距离
    DISTANCE_H = [dist_min, dist_max, dist_ave] # 豪斯多夫距离
    DIST_NAME = ['single linkage', 'complete linkage', 'average linkage']
    X, Y = generate_samples(count, centers, std)

    for dist in DISTANCE_E:
        C, IndexC = AGNES(X, Y, dist, k)
        y_hat = np.zeros(count, dtype=int)
        for i in range(len(IndexC)):
            y_hat[np.array(IndexC[i])] = i
        
        creat_plot(X, Y, y_hat, k, DIST_NAME[DISTANCE_E.index(dist)])
