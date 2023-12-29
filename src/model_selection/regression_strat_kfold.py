import warnings
from abc import ABCMeta

from sklearn.model_selection import StratifiedKFold

from src.model_selection.utils import reg_classification
from src.visualization.plot3d import scatter3d


class RegressionStratKFold(StratifiedKFold, metaclass=ABCMeta):
    def __init__(self,
                 strat_col_index=-1,
                 n_clusters=6,
                 **args):
        self.strat_col_indx = strat_col_index
        self.n_clusters = n_clusters

        self._features = None

        super().__init__(**args)

    def split(self, X, y=None, groups=None):
        clusters, features = reg_classification(X, strat_col_indx=self.strat_col_indx, n_clusters=self.n_clusters)
        self._features = features

        scatter3d(features, axe_titles=('a', 'b', 'std'), colors=clusters)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return super().split(X, clusters)
