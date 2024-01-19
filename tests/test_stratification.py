from src.model_selection.regression_strat_kfold import RegressionStratKFold
from src.model_selection.regression_train_test_split import RegressionTrainTestSplitter
from src.session.model_manager import _get_sequences

import numpy as np
import pandas as pd

from src.visualization.plot3d import scatter3d

CSV_PATH = '../data/input.csv'


def test_kfold(sequences, strat_col_indx):
    kfold = RegressionStratKFold(n_clusters=7, n_splits=5, strat_col_index=strat_col_indx)
    splits = kfold.split(sequences)

    splits_list = []
    splits_classes = []
    for split_index, (_, test_idx) in enumerate(splits):
        for i in test_idx:
            splits_list.append(kfold.features[i])
            splits_classes.append(split_index)

    splits_list = np.stack(splits_list)
    splits_classes = np.array(splits_classes)

    scatter3d(features=splits_list, colors=splits_classes)


def test_train_test_split(sequences, strat_col_indx):
    # Perform split
    splitter = RegressionTrainTestSplitter()
    _, _ = splitter.fit_split(X=sequences, test_size=0.25, n_clusters=7, strat_col_index=strat_col_indx)

    # Extract trajectory features and indices of train and test
    features = splitter._features
    train_indices = splitter._train_indices
    test_indices = splitter._test_indices

    entries = []
    classes = [0 for _ in train_indices]
    classes.extend([1 for _ in test_indices])

    # Get features that correspond to train elements
    for i in train_indices:
        entry = features[i]
        entries.append(entry)

    # Get features that correspond to test elements
    for i in test_indices:
        entry = features[i]
        entries.append(entry)

    entries = np.stack(entries)

    scatter3d(features=features, colors=splitter._clusters, axe_titles=('a', 'b', 'std'))
    scatter3d(features=entries, colors=np.array(classes), axe_titles=('a', 'b', 'std'))

    pass


if __name__ == '__main__':
    _sequences, _ = _get_sequences(CSV_PATH)
    _strat_col_indx = -1

    test_kfold(_sequences, _strat_col_indx)
    # test_train_test_split(_sequences, _strat_col_indx)

    pass
