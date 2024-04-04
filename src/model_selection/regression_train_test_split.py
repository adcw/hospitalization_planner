from typing import Optional, List

import numpy as np
from sklearn.model_selection import train_test_split

from src.model_selection.utils import reg_classification, stratify_classes
from src.visualization.plot3d import scatter3d


class RegressionTrainTestSplitter:

    def __init__(self):
        self._features = None
        self._clusters = None
        self._train_indices = None
        self._test_indices = None

    def fit_split(self,
                  X: list[np.iterable],
                  test_size: float = 0.1,
                  strat_col_index: int = -1,
                  n_clusters: int = 5
                  ):
        self._clusters, self._features = reg_classification(X,
                                                            strat_col_indx=strat_col_index,
                                                            n_clusters=n_clusters)

        self._train_indices, self._test_indices = stratify_classes(self._clusters,
                                                                   test_size=test_size)

        return [X[i] for i in self._train_indices], [X[i] for i in self._test_indices]

    def plot_split(self,
                   title: str = "Train and test split",
                   axe_titles: Optional[List[str]] = None
                   ):
        axe_titles = axe_titles or ['a', 'b', 'std']

        train_features = self._features[self._train_indices]
        test_features = self._features[self._test_indices]

        scatter3d(features=np.concatenate([train_features, test_features], axis=0),
                  colors=np.array([0 for _ in train_features] + [1 for _ in test_features]),
                  title=title,
                  axe_titles=axe_titles
                  )
