from src.model_selection.stratification import RegressionStratKFold
from src.session.model_manager import _get_sequences

import numpy as np

from src.visualization.plot3d import scatter3d

CSV_PATH = '../data/input.csv'

if __name__ == '__main__':
    sequences, _ = _get_sequences(CSV_PATH)
    strat_col_indx = -1
    strat_colname = sequences[0].columns.values[strat_col_indx]

    kfold = RegressionStratKFold(n_clusters=7, n_splits=5, strat_col_indx=strat_col_indx)
    splits = kfold.split(sequences)

    splits_list = []
    splits_classes = []
    for split_index, (_, test_idx) in enumerate(splits):
        for i in test_idx:
            splits_list.append(kfold._features[i])
            splits_classes.append(split_index)

    splits_list = np.stack(splits_list)
    splits_classes = np.array(splits_classes)

    scatter3d(features=splits_list, colors=splits_classes)

    pass
