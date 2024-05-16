from typing import List, Optional

import pandas as pd
import torch

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
WINDOW_SIZE = 10
STRIDE_RATE = 0.001
N_CLASSES = 6
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


def generate_breathing_dataset(csv_path: str,
                               usecols: List[str],

                               window_size: int,
                               stride_rate: float,

                               n_classes: int,

                               test_perc: float,

                               pattern_cluster_cols: List[str],
                               limit: Optional[int] = None,
                               ):
    """
    Generuje dane, przeprowadza próbkowanie, skaluje, tworzy okna i klastry oraz wizualizuje reguły klastrów.

    Args:
        csv_path (str): Ścieżka do pliku CSV zawierającego dane.
        usecols (List[str]): Lista kolumn do wykorzystania z pliku CSV.
        window_size (int): Rozmiar okna.
        stride_rate (float): Współczynnik kroku.
        n_classes (int): Liczba klastrów do utworzenia.
        max_tree_depth (int): Maksymalna głębokość drzewa decyzyjnego.
        test_perc (float): Procent danych przeznaczonych do testów.
        pattern_cluster_cols (List[str]): Lista kolumn do wykorzystania do tworzenia klastrów.

    Returns:
        None
    """
    sequences, preprocessor = _get_sequences(path=csv_path, limit=limit, usecols=usecols)

    labels = label_sequences(sequences, stratify_cols=PATTERN_CLUSTER_COLS).labels_

    sequences_train, sequences_test = train_test_split_safe(sequences, stratify=labels, test_size=test_perc)

    sequences_train_scaled, scaler = scale(sequences_train)

    # Make windows and cluster them
    print("Preparing windows for clustering...")
    windows = make_windows(sequences_train_scaled, window_size=window_size,
                           stride=max(1, round(window_size * stride_rate)))

    print("Performing clustering...")
    kmed = learn_clusters(windows, n_clusters=n_classes, input_cols=pattern_cluster_cols)

    # Unscale windows and visualize clustering rules with the use of tree
    original_windows = [pd.DataFrame(scaler.inverse_transform(w), columns=w.columns) for w in
                        windows]

    print("Discovering cluster rules...")
    original_w_features = visualize_clustering_rules(original_windows, labels=kmed.labels_,
                                                     input_cols=pattern_cluster_cols)

    input("Proceed?")

    # Show cluster centers
    plot_data = []

    for i, med_i in enumerate(kmed.medoid_indices_):
        plot_data.append({
            'class': i,  # dtype: int
            'features': original_w_features.iloc[med_i, :],  # dtype: pd.Series
            'window': original_windows[med_i]  # dtype: pd.DataFrame
        })

    plot_medoid_data(plot_data, PATTERN_CLUSTER_COLS)

    # Creating windows for training
    print("Preparing windows for clustering...")
    windows = make_windows(sequences_train_scaled, window_size=window_size * 2,
                           stride=max(1, round(window_size * stride_rate)))

    x_windows, y_windows = xy_windows_split(windows, target_len=window_size, min_x_len=int(0.5 * window_size))

    y_window_features = extract_seq_features(y_windows, input_cols=pattern_cluster_cols)
    ys_classes = kmed.predict(y_window_features)

    x_window_features = extract_seq_features(x_windows, input_cols=pattern_cluster_cols)
    xs_classes = kmed.predict(x_window_features)

    xs = [torch.Tensor(x.values) for x in x_windows]
    xs = pad_left(xs, window_size=window_size)

    ds = BreathingDataset(xs=xs,
                          xs_classes=xs_classes,
                          ys_classes=ys_classes,
                          test_sequences=sequences_test,
                          window_size=WINDOW_SIZE,
                          kmed=kmed,
                          scaler=scaler,
                          )
    curr_path = base_dir()
    ds.save(f"{curr_path}/breathing_dataset.pkl")

    print("Done!")

    pass


if __name__ == '__main__':
    set_seed()
    run_path = get_run_path("../../../bp_dataset_creation_runs")
    print(f"Current run path: {run_path}")
    base_dir(run_path)

    generate_breathing_dataset(csv_path=CSV_PATH,
                               usecols=COLS,
                               window_size=WINDOW_SIZE,
                               stride_rate=STRIDE_RATE,
                               n_classes=N_CLASSES,
                               test_perc=TEST_PERC,
                               pattern_cluster_cols=PATTERN_CLUSTER_COLS,

                               limit=None
                               )

    pass
