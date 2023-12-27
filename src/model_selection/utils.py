import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_features(seq: np.ndarray) -> np.ndarray:
    reg = LinearRegression()
    lsp = np.linspace(0, len(seq) - 1, len(seq))

    reg.fit(lsp.reshape(-1, 1), seq)
    dev = np.var(seq)

    return np.array([reg.coef_[0], reg.intercept_, dev])


def reg_classification(
        X: list[np.iterable],
        strat_col_indx: int = -1,
        n_clusters: int = 5
):
    n_cols = X[0].shape[1]
    real_index = (n_cols + strat_col_indx) % n_cols

    if type(X[0]) == pd.DataFrame:
        X = [x.values for x in X]

    features = [get_features(s[:, 16]) for s in X]
    features = np.stack(features)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    clusters = kmeans.fit_predict(features_scaled)

    return clusters, features


def stratify_classes(classes, test_size=0.2, random_state=None):
    if type(classes) == list:
        y = np.array(classes)

    unique_labels = np.unique(classes)
    train_indices, test_indices = [], []

    for label in unique_labels:
        label_indices = np.where(classes == label)[0]

        if label_indices.shape[0] == 1:
            label_indices = np.concatenate([label_indices, label_indices])

        train_idx, test_idx = train_test_split(label_indices, test_size=test_size, random_state=random_state)
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)

    return train_indices, test_indices


if __name__ == '__main__':
    _X = [10, 20, 30, 40, 50, 60, 70, 80]
    _y = [0, 0, 1, 1, 1, 1, 2, 2]

    train_indices, test_indices = stratify_classes(_y, test_size=0.5)
    pass
