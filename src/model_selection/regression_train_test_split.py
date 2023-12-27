import numpy as np

from src.model_selection.utils import reg_classification, stratify_classes


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
                  n_clusters: int = 5,
                  random_state=None
                  ):
        self._clusters, self._features = reg_classification(X,
                                                            strat_col_indx=strat_col_index,
                                                            n_clusters=n_clusters)

        self._train_indices, self._test_indices = stratify_classes(self._clusters,
                                                                   test_size=test_size,
                                                                   random_state=random_state)

        # TODO: Plot split
        # train_features = self._features[self._train_indices]
        #
        #
        # scatter3d(features=splits_list, colors=splits_classes)

        return [X[i] for i in self._train_indices], [X[i] for i in self._test_indices]
