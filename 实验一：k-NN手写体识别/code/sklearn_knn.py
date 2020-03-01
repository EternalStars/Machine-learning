from sklearn import neighbors
import numpy as np
import knn

def sklearn_knn(K):
    # 加载数据
    test = knn.load_data('data/semeion_test.csv')
    train = knn.load_data('data/semeion_train.csv')
    # 数据集切片
    train_data, train_labels = train[:, :-1], train[:, -1]
    test_data, test_labels = test[:, :-1], test[:, -1]
    # 初始化
    k_max = K
    train_accuracy = np.empty(k_max)
    test_accuracy = np.empty(k_max)

    for k in range(0, k_max):
        sknn = neighbors.KNeighborsClassifier(n_neighbors=k+1)
        sknn.fit(train_data, train_labels)
        train_accuracy[k] = sknn.score(train_data, train_labels)
        test_accuracy[k] = sknn.score(test_data, test_labels)
    print("train accuracy:", train_accuracy, "\ntest accuracy:", test_accuracy)
    return train_accuracy, test_accuracy
