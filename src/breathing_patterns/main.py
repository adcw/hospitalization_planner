import random
from typing import List, Optional

import numpy as np

import dtreeviz
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn_extra.cluster import KMedoids

import data.colnames_original as c
from src.config.seeds import set_seed
from src.error_analysis.extract import extract_seq_features
from src.model_selection.stratified_sampling import stratified_sampling
from src.session.model_manager import _get_sequences
from src.session.utils.save_plots import save_viz, base_dir
from src.tools.dataframe_scale import scale
from src.tools.iterators import windowed
from src.visualization.plot_clusters import visualize_clusters

from data.chosen_colnames import COLS

CSV_PATH = './../../data/input.csv'
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


def make_windows(seqs: List[pd.DataFrame], window_size: int, stride: int):
    windows = []

    for seq in seqs:
        windows.extend([w for w in windowed(seq, window_size=window_size, stride=stride)])

    return windows


def xy_windows_split(windows: List, target_len: int, min_x_len: Optional[int] = None):
    if min_x_len is None:
        min_x_len = target_len

    x_windows = []
    y_windows = []
    for w in windows:
        w_len = len(w)
        if w_len < min_x_len + target_len:
            continue

        x_windows.append(w[:-target_len])
        y_windows.append(w[-target_len:])

    return x_windows, y_windows


def train_test_split(seqs: List[pd.DataFrame], stratify_cols: Optional[List[str]], test_perc: float):
    seq_features = extract_seq_features(seqs, input_cols=stratify_cols)
    kmed_split = KMedoids(n_clusters=min(10, len(seqs)))
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


def learn_clusters(windows: List[pd.DataFrame], n_clusters: int, input_cols: Optional[List[str]] = None):
    features = extract_seq_features(windows, input_cols=input_cols)
    kmed = KMedoids(n_clusters=n_clusters)
    kmed.fit(features)

    visualize_clusters(features, kmed.labels_, kmed.cluster_centers_)

    return kmed


def visualize_clustering_rules(windows: List[pd.DataFrame], labels: List,
                               max_tree_depth: int = 4, input_cols: Optional[List[str]] = None):
    features = extract_seq_features(windows, input_cols=input_cols)
    clf = DecisionTreeClassifier(max_depth=max_tree_depth).fit(features, labels)

    viz_model = dtreeviz.model(clf,
                               X_train=features,
                               feature_names=features.columns,

                               y_train=labels,

                               target_name="Klasy")

    viz = viz_model.view()
    save_viz("pattern_tree.svg", viz)


if __name__ == '__main__':
    set_seed()
    base_dir("./clustering_runs")

    patter_cluster_cols = [col for col in PATTERN_CLUSTER_COLS if col in COLS]

    print("Reading data")
    sequences, preprocessor = _get_sequences(path=CSV_PATH, limit=None, usecols=COLS)
    sequences_train, sequences_test = train_test_split(random.choices(sequences, k=len(sequences)),
                                                       stratify_cols=[c.RESPIRATION], test_perc=TEST_PERC)

    sequences_train_scaled, scaler = scale(sequences_train)

    # Make windows and cluster them
    windows = make_windows(sequences_train_scaled, window_size=WINDOW_SIZE,
                           stride=max(1, round(WINDOW_SIZE * STRIDE_RATE)))

    kmed = learn_clusters(windows, n_clusters=N_CLASSES, input_cols=patter_cluster_cols)

    # Unscale windows and visualize clustering rules with the use of tree
    original_windows = [pd.DataFrame(scaler.inverse_transform(w), columns=w.columns) for w in
                        windows]

    visualize_clustering_rules(original_windows, labels=kmed.labels_, max_tree_depth=MAX_TREE_DEPTH,
                               input_cols=patter_cluster_cols)

    pass
