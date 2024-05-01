import random
from typing import List

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

CSV_PATH = './../../data/input.csv'
WINDOW_SIZE = 15
STRIDE_RATE = 0.2
N_CLASSES = 7
MAX_TREE_DEPTH = 5
TRAIN_PERC = 0.8

COLS = [
    c.PATIENTID,
    c.DATEID,

    c.PO2,
    c.ANTYBIOTYK,
    c.RTG_RDS,
    c.RTG_PDA,
    c.DOPAMINA,
    c.PTL,
    c.STERYD,
    c.FIO2,
    c.GENERAL_SURFACTANT,
    c.RESPIRATION,
]


def make_windows(seqs: List[pd.DataFrame], stride: int):
    windows = []

    for seq in seqs:
        windows.extend([w for w in windowed(seq, window_size=WINDOW_SIZE, stride=stride)])

    return windows


def train_test_split(seqs: List[pd.DataFrame]):
    seq_features = extract_seq_features(seqs)
    kmed_split = KMedoids(n_clusters=min(10, len(seqs)))
    kmed_split.fit(seq_features)

    train_indices = stratified_sampling(kmed_split.labels_, sample_size=round(TRAIN_PERC * len(seqs)))

    sequences_train = []
    sequences_test = []

    for i in range(len(seqs)):
        if i in train_indices:
            sequences_train.append(seqs[i])
        else:
            sequences_test.append(seqs[i])

    return sequences_train, sequences_test


if __name__ == '__main__':
    set_seed()
    base_dir("./clustering_runs")

    print("Reading data")
    sequences, preprocessor = _get_sequences(path=CSV_PATH, limit=400, usecols=COLS)

    sequences_scaled, scaler = scale(sequences)

    # Make windows
    windows_from_scaled = make_windows(sequences_scaled, stride=max(1, round(WINDOW_SIZE * STRIDE_RATE)))

    # Create clusters
    features_from_scaled = extract_seq_features(windows_from_scaled)

    print("Creating clusters")
    kmed_patterns = KMedoids(n_clusters=N_CLASSES)
    kmed_patterns.fit(features_from_scaled)

    visualize_clusters(features_from_scaled, kmed_patterns.labels_, kmed_patterns.cluster_centers_)

    # Visualize cluster rules with tree
    print("Extracting cluster rules")
    windows_from_unscaled = [pd.DataFrame(scaler.inverse_transform(w), columns=w.columns) for w in windows_from_scaled]
    features_from_unscaled = extract_seq_features(windows_from_unscaled)

    clf = DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH).fit(features_from_unscaled, kmed_patterns.labels_)

    viz_model = dtreeviz.model(clf,
                               X_train=features_from_unscaled,
                               feature_names=features_from_unscaled.columns,

                               y_train=kmed_patterns.labels_,

                               target_name="Klasy")

    viz = viz_model.view()
    save_viz("pattern_tree.svg", viz)

    # Create training data
    print("Creating data")

    sequences_train, sequences_test = train_test_split(sequences_scaled)


    pass
