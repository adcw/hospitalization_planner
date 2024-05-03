import logging
import os.path
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn_extra.cluster import KMedoids

import data.colnames_original as c
from data.chosen_colnames import COLS
from src.breathing_patterns.data.dataset import BreathingDataset
from src.breathing_patterns.utils.clustering import learn_clusters, visualize_clustering_rules
from src.breathing_patterns.utils.plot import plot_medoid_data
from src.breathing_patterns.utils.windows import make_windows, xy_windows_split
from src.config.seeds import set_seed
from src.error_analysis.extract import extract_seq_features
from src.model_selection.stratified_sampling import stratified_sampling
from src.session.model_manager import _get_sequences
from src.session.utils.save_plots import base_dir
from src.tools.dataframe_scale import scale

CSV_PATH = '../../../data/input.csv'
WINDOW_SIZE = 10
STRIDE_RATE = 0.2
N_CLASSES = 7
MAX_TREE_DEPTH = 5
TEST_PERC = 0.1

# Columns used to extract breathing patterns
PATTERN_CLUSTER_COLS = [
    c.PO2,
    # c.ANTYBIOTYK,
    c.RTG_RDS,
    # c.RTG_PDA,
    # c.DOPAMINA,
    # c.PTL,
    # c.STERYD,
    c.FIO2,
    c.GENERAL_SURFACTANT,
    c.RESPIRATION,
]


def train_test_split(seqs: List[pd.DataFrame], stratify_cols: Optional[List[str]], test_perc: float):
    seq_features = extract_seq_features(seqs, input_cols=stratify_cols)
    kmed_split = KMedoids(n_clusters=min(10, len(seqs)), init='k-medoids++')
    kmed_split.fit(seq_features)

    test_indices = stratified_sampling(kmed_split.labels_, sample_size=round(test_perc * len(seqs)))

    seqs_train = []
    seqs_test = []

    for i in range(len(seqs)):
        if i in test_indices:
            seqs_test.append(seqs[i])
        else:
            seqs_train.append(seqs[i])

    return seqs_train, seqs_test


def generate_breathing_dataset(csv_path: str,
                               usecols: List[str],

                               window_size: int,
                               stride_rate: float,

                               n_classes: int,
                               max_tree_depth: int,

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
    sequences_train, sequences_test = train_test_split(random.choices(sequences, k=len(sequences)),
                                                       stratify_cols=[c.RESPIRATION], test_perc=test_perc)

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
                                                     max_tree_depth=max_tree_depth,
                                                     input_cols=pattern_cluster_cols)

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
    ys = kmed.predict(y_window_features)

    ds = BreathingDataset(xs=x_windows, ys=ys)
    curr_path = base_dir()
    ds.save(f"{curr_path}/breathing_dataset.pkl")

    pass


def get_run_path(path: str):
    abs_path = os.path.abspath(path)

    os.makedirs(path)

    contents = [os.path.join(abs_path, f) for f in os.listdir(abs_path)]
    run_dirs = [d for d in contents if os.path.isdir(d)]

    curr_index = 1
    max_index = 0
    if len(run_dirs) > 0:
        for r in run_dirs:
            try:
                n = int(r.split("\\")[-1].split("_")[-1])
                if n > max_index:
                    max_index = n
            except ValueError:
                pass
        curr_index = max_index + 1

    return os.path.join(abs_path, f"run_{curr_index}")


if __name__ == '__main__':
    set_seed()
    run_path = get_run_path("../../../bp_dataset_creation_runs")
    base_dir(run_path)

    generate_breathing_dataset(csv_path=CSV_PATH,
                               usecols=COLS,
                               window_size=WINDOW_SIZE,
                               stride_rate=STRIDE_RATE,
                               n_classes=N_CLASSES,
                               max_tree_depth=MAX_TREE_DEPTH,
                               test_perc=TEST_PERC,
                               pattern_cluster_cols=PATTERN_CLUSTER_COLS,

                               limit=None
                               )

    pass
