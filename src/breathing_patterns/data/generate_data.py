from typing import List, Optional

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src.breathing_patterns.data.dataset import BreathingDataset
from src.breathing_patterns.pattern_cluster_cols import PATTERN_CLUSTER_COLS
from src.breathing_patterns.utils.clustering import learn_clusters, visualize_clustering_rules, label_sequences
from src.breathing_patterns.utils.plot import plot_medoid_data
from src.breathing_patterns.utils.windows import make_windows, xy_windows_split
from src.chosen_colnames import COLS
from src.config.seeds import set_seed
from src.model_selection.stratified import train_test_split_safe
from src.session.model_manager import _get_sequences
from src.session.utils.save_plots import base_dir
from src.tools.dataframe_scale import scale
from src.tools.extract import extract_seq_features
from src.tools.run_utils import get_run_path

CSV_PATH = '../../../data/input.csv'

PATTERN_WINDOW_SIZE = 3
HISTORY_WINDOW_SIZE = 10

STRIDE_RATE = 0.001
N_CLASSES = 5
TEST_PERC = 0.2


def pad_left(tensor_list, window_size):
    """
    Pads tensors in the list with zeros on the left side if their first dimension is shorter than window_size.

    Args:
    - tensor_list: list of PyTorch tensors
    - window_size: int, target size for the first dimension of tensors

    Returns:
    - padded_list: list of PyTorch tensors with the same shape as input tensors but padded on the left side
    """
    padded_list = []
    for tensor in tensor_list:
        pad_size = window_size - tensor.size(0)
        if pad_size > 0:
            pad = torch.zeros((pad_size,) + tensor.size()[1:], dtype=tensor.dtype, device=tensor.device)
            padded_tensor = torch.cat((pad, tensor), dim=0)
        else:
            padded_tensor = tensor
        padded_list.append(padded_tensor)
    return padded_list


def extract_patterns(
        sequences: List[pd.DataFrame],
        pattern_window_size: int,
        stride_rate: float,
        n_classes: int,
        pattern_cluster_cols: List[str],
):
    # Make windows and cluster them
    print("Preparing windows for clustering...")
    windows = make_windows(sequences, window_size=pattern_window_size,
                           stride=max(1, round(pattern_window_size * stride_rate)))

    print("Performing clustering...")
    kmed = learn_clusters(windows, n_clusters=n_classes, input_cols=pattern_cluster_cols)

    # Unscale windows and visualize clustering rules with the use of tree
    # original_windows = [pd.DataFrame(scaler.inverse_transform(w), columns=w.columns) for w in
    #                     windows]

    print("Discovering cluster rules...")
    original_w_features = visualize_clustering_rules(windows, labels=kmed.labels_,
                                                     input_cols=pattern_cluster_cols)

    # Show cluster centers
    plot_data = []

    for i, med_i in enumerate(kmed.medoid_indices_):
        plot_data.append({
            'class': i,  # dtype: int
            'features': original_w_features.iloc[med_i, :],  # dtype: pd.Series
            'window': windows[med_i]  # dtype: pd.DataFrame
        })

    plot_medoid_data(plot_data, PATTERN_CLUSTER_COLS)

    return kmed


def get_dataset(
        sequences_train: List[pd.DataFrame],
        sequences_test: List[pd.DataFrame],

        pattern_window_size: int,
        history_window_size: int,

        stride_rate: float,

        pattern_cluster_cols: List[str],

        kmed,
):
    sequences_train, scaler = scale(sequences_train)

    # Creating windows for training
    print("Preparing windows for clustering...")
    train_windows = make_windows(sequences_train, window_size=history_window_size + pattern_window_size,
                                 stride=max(1, round(history_window_size * stride_rate)))

    x_windows, y_windows = xy_windows_split(train_windows, target_len=pattern_window_size,
                                            min_x_len=int(0.5 * history_window_size))

    y_window_features = extract_seq_features(y_windows, input_cols=pattern_cluster_cols)
    ys_classes = kmed.predict(y_window_features)

    x_window_features = extract_seq_features(x_windows, input_cols=pattern_cluster_cols)
    xs_classes = kmed.predict(x_window_features)

    xs = [torch.Tensor(x.values) for x in x_windows]
    xs = pad_left(xs, window_size=history_window_size)

    print("Done!")

    return BreathingDataset(xs=xs,
                            xs_classes=xs_classes,
                            ys_classes=ys_classes,
                            test_sequences=sequences_test,

                            pattern_window_size=pattern_window_size,
                            history_window_size=history_window_size,

                            kmed=kmed,
                            scaler=scaler
                            )


def generate_breathing_dataset(sequences_train: List[pd.DataFrame],
                               sequences_test: List[pd.DataFrame],

                               pattern_window_size: int,
                               history_window_size: int,

                               stride_rate: float,

                               n_classes: int,

                               pattern_cluster_cols: List[str],
                               ):
    kmed = extract_patterns(
        sequences=[*sequences_train, *sequences_test],
        n_classes=n_classes,
        stride_rate=stride_rate,
        pattern_window_size=pattern_window_size,
        pattern_cluster_cols=pattern_cluster_cols,
    )

    return get_dataset(
        sequences_train=sequences_train,
        sequences_test=sequences_test,

        pattern_window_size=pattern_window_size,
        history_window_size=history_window_size,

        pattern_cluster_cols=pattern_cluster_cols,
        stride_rate=stride_rate,

        kmed=kmed
    )


if __name__ == '__main__':
    set_seed()
    run_path = get_run_path("../../../bp_dataset_creation_runs")
    print(f"Current run path: {run_path}")
    base_dir(run_path)

    _sequences, preprocessor = _get_sequences(path=CSV_PATH, usecols=COLS)
    # _sequences, _scaler = scale(_sequences)
    labels = label_sequences(_sequences, stratify_cols=PATTERN_CLUSTER_COLS).labels_

    _sequences_train, _sequences_test = train_test_split_safe(_sequences, stratify=labels, test_size=TEST_PERC)

    _ds = generate_breathing_dataset(
        sequences_train=_sequences_train,
        sequences_test=_sequences_test,

        pattern_window_size=PATTERN_WINDOW_SIZE,
        history_window_size=HISTORY_WINDOW_SIZE,

        stride_rate=STRIDE_RATE,
        n_classes=N_CLASSES,
        pattern_cluster_cols=PATTERN_CLUSTER_COLS,
    )

    _ds.save(f"{run_path}/breathing_dataset.pkl")

    pass
