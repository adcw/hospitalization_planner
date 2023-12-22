import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.visualization.plot3d import scatter3d


def get_features(seq: np.ndarray) -> np.ndarray:
    reg = LinearRegression()
    lsp = np.linspace(0, len(seq) - 1, len(seq))

    reg.fit(lsp.reshape(-1, 1), seq)
    dev = np.var(seq)

    return np.array([reg.coef_[0], reg.intercept_, dev])


class RegressionStratKFold(StratifiedKFold):
    def __init__(self,
                 strat_col_indx=-1,
                 n_clusters=6,
                 **args):
        self.strat_col_indx = strat_col_indx
        self.n_clusters = n_clusters

        self._features = None

        super().__init__(**args)

    def split(self, X, y=None, groups=None):
        n_cols = X[0].shape[1]
        real_index = (n_cols + self.strat_col_indx) % n_cols
        colname = X[0].columns.values[real_index]

        features = [get_features(s[colname]) for s in X]
        features = np.stack(features)
        self._features = features

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        clusters = kmeans.fit_predict(features_scaled)

        scatter3d(features, labels=('a', 'b', 'std'), colors=clusters)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return super().split(X, clusters)
