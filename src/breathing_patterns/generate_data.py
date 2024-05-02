import random
from typing import List, Optional

import pandas as pd
from sklearn_extra.cluster import KMedoids

import data.colnames_original as c
from data.chosen_colnames import COLS
from src.breathing_patterns.utils.clustering import learn_clusters, visualize_clustering_rules
from src.breathing_patterns.utils.windows import make_windows
from src.config.seeds import set_seed
from src.error_analysis.extract import extract_seq_features
from src.model_selection.stratified_sampling import stratified_sampling
from src.session.model_manager import _get_sequences
from src.session.utils.save_plots import base_dir
from src.tools.dataframe_scale import scale

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


def generate_data(csv_path: str,
                  usecols: List[str],

                  window_size: int,
                  stride_rate: float,

                  n_classes: int,
                  max_tree_depth: int,

                  test_perc: float,

                  pattern_cluster_cols: List[str]
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
    sequences, preprocessor = _get_sequences(path=csv_path, limit=None, usecols=usecols)
    sequences_train, sequences_test = train_test_split(random.choices(sequences, k=len(sequences)),
                                                       stratify_cols=[c.RESPIRATION], test_perc=test_perc)

    sequences_train_scaled, scaler = scale(sequences_train)

    # Make windows and cluster them
    windows = make_windows(sequences_train_scaled, window_size=window_size,
                           stride=max(1, round(window_size * stride_rate)))

    kmed = learn_clusters(windows, n_clusters=n_classes, input_cols=pattern_cluster_cols)

    # Unscale windows and visualize clustering rules with the use of tree
    original_windows = [pd.DataFrame(scaler.inverse_transform(w), columns=w.columns) for w in
                        windows]

    visualize_clustering_rules(original_windows, labels=kmed.labels_, max_tree_depth=max_tree_depth,
                               input_cols=pattern_cluster_cols)


if __name__ == '__main__':
    set_seed()
    base_dir("./clustering_runs")

    generate_data(csv_path=CSV_PATH,
                  usecols=COLS,
                  window_size=WINDOW_SIZE,
                  stride_rate=STRIDE_RATE,
                  n_classes=N_CLASSES,
                  max_tree_depth=MAX_TREE_DEPTH,
                  test_perc=TEST_PERC,
                  pattern_cluster_cols=PATTERN_CLUSTER_COLS)
