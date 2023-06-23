import numpy as np
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier


def train_mlp(X_train, y_train, scaler=None):
    print(X_train.shape)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))

    # X_train = scaler.transform(X_train)
    clf = MLPClassifier(max_iter=1000).fit(X_train, y_train)
    return clf


def predict_mlp(clf, X_data, scaler=None):
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1] * X_data.shape[2]))
    # X_data = scaler.transform(X_data)
    pred = clf.predict(X_data)
    y_data = pred
    return y_data


def predict_threshold_mlp(clf, X_data, threshold=0.7, scaler=None):
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1] * X_data.shape[2]))
    # X_data = scaler.transform(X_data)
    pred = clf.predict_proba(X_data)
    survived_index = []
    y_data = []
    for i, p in enumerate(pred):
        if p.max() >= threshold:
            survived_index.append(i)
            y_data.append(p.argmax())
    return np.array(y_data), np.array(survived_index)


from sklearn.metrics import pairwise_distances


def train_knn(X_train, y_train, k=11):
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    clf = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    return clf


def predict_knn(clf, X_data):
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1] * X_data.shape[2]))
    y_data = clf.predict(X_data)
    return y_data


# def predict_threshold_knn(X_data, y_data, X_target, K=11, threshold=9):
#     X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1] * X_data.shape[2]))
#     X_target = np.reshape(X_target, (X_target.shape[0], X_target.shape[1] * X_target.shape[2]))
#     distances = pairwise_distances(X_target, X_data, metric='euclidean')
#     survived_idx = []
#     y_target = []
#     for i in range(X_target.shape[0]):
#         sorted_row_idx = np.argsort(distances[i])
#         labels_row = y_data[sorted_row_idx[:K]]
#         val, counts = np.unique(labels_row, return_counts=True)
#         # print(counts.max())
#         if counts.max() >= threshold:
#             y_row = val[counts.argmax()]
#             y_target.append(y_row)
#             survived_idx.append(i)
#
#     # print(survived_idx)
#     return np.array(y_target), np.array(survived_idx)

from sklearn.neighbors import NearestNeighbors


def predict_threshold_knn(X_data, y_data, X_target, k=11, threshold=9):
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1] * X_data.shape[2]))
    X_target = np.reshape(X_target, (X_target.shape[0], X_target.shape[1] * X_target.shape[2]))
    survived_idx = []
    y_target = []
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X_data)
    neighbors_dist_target, neighbors_target = neigh.kneighbors(X_target)
    for i in range(neighbors_target.shape[0]):
        labels_row = y_data[neighbors_target[i]]
        val, counts = np.unique(labels_row, return_counts=True)
        if counts.max() >= threshold:
            y_row = val[counts.argmax()]
            y_target.append(y_row)
            survived_idx.append(i)
    return np.array(y_target), np.array(survived_idx)
