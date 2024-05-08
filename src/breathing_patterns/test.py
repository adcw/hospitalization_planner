from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.breathing_patterns.data.dataset import BreathingDataset
from src.breathing_patterns.models.BreathingPatterModel import BreathingPatternModel
from src.breathing_patterns.patter_cluster_cols import PATTERN_CLUSTER_COLS
from src.breathing_patterns.utils.clustering import learn_clusters
from src.config.seeds import set_seed
from src.tools.extract import extract_seq_features
from src.model_selection.stratified import stratified_sampling
from src.session.utils.save_plots import base_dir, save_plot
from src.tools.iterators import windowed
from src.tools.reports import save_report_and_conf_m
from src.tools.run_utils import get_run_path

RESPIRATION = 'respiration'
GENERAL_SURFACTANT = 'GENERAL_SURFACTANT'
FIO2 = 'FiO2'
RTG_RDS = 'RTG_RDS'
PO2 = 'po2'
ANTYBIOTYK = 'ANTYBIOTYK'
STERYD = 'STERYD'

DATASET_PATH = "../../bp_dataset_creation_runs/run_2/breathing_dataset.pkl"
RUN_PATH = "../../bp_test_runs"
MODEL_PATH = "../../bp_train_runs/run_3/model.pkl"

MAX_PLOTS = 10
MIN_X_WINDOW_RATE = 0.8
STRIDE_RATE = 0.2


def plot_breathing(
        df: pd.DataFrame,
        ranges: List[List],
        y_labels_true: List[int],
        y_labels_pred: List[int],
        n_classes: int,
        plot_index: Optional[int] = None
):
    stride = ranges[1][0] - ranges[0][0] if len(ranges) > 1 else 1
    w_len = len(ranges[0])

    n_lists = (int(np.ceil(w_len / stride)) + 1) if len(ranges) > 1 else 1

    ranges_stairs = [[] for _ in range(n_lists)]
    real_labels_stairs = [[] for _ in range(n_lists)]
    pred_labels_stairs = [[] for _ in range(n_lists)]

    for i, r in enumerate(ranges):
        ranges_stairs[i % n_lists].append([r[int(w_len / 2)], r[-1]])
        real_labels_stairs[i % n_lists].append(y_labels_true[i])
        pred_labels_stairs[i % n_lists].append(y_labels_pred[i])

    n_colors = n_classes
    colors = matplotlib.colormaps['viridis'].resampled(n_colors)([x for x in range(n_colors)])

    fig, ax = plt.subplots(len(df.columns) + 1, 1, figsize=(10, 14), sharex=True,
                           gridspec_kw={'hspace': 0.5, 'height_ratios': [*([1] * len(df.columns)), 0.2 * n_lists]})

    for i, column in enumerate(df.columns):
        ax[i].plot(df[column])
        ax[i].set_title(column)

    for row_i, row in enumerate(ranges_stairs):
        for rng_i, rng in enumerate(row):
            real_label_color = colors[real_labels_stairs[row_i][rng_i]]
            pred_label_color = colors[pred_labels_stairs[row_i][rng_i]]

            ax[-1].hlines((len(ranges_stairs) - row_i) * 3, xmin=rng[0], xmax=rng[-1], linewidth=5,
                          color=real_label_color)
            ax[-1].hlines((len(ranges_stairs) - row_i) * 3 - 1, xmin=rng[0], xmax=rng[-1], linewidth=5,
                          color=pred_label_color)
            ax[-1].set_yticks([])

    fig.suptitle("Breathing parameters")
    save_plot(f"cases/{plot_index or 'case'}.png")


if __name__ == '__main__':
    set_seed()
    run_path = get_run_path(RUN_PATH)
    print(f"Current run path: {run_path}")
    base_dir(run_path)

    bd = BreathingDataset.read(DATASET_PATH)
    model = BreathingPatternModel()
    model.load(MODEL_PATH)

    sequences = bd.test_sequences
    scaled_sequences = [pd.DataFrame(bd.scaler.transform(s), columns=s.columns) for s in sequences]
    kmed = learn_clusters(scaled_sequences, n_clusters=12, input_cols=PATTERN_CLUSTER_COLS, save_plots=False)

    # Perform quick test
    to_chose = min(MAX_PLOTS, len(bd.test_sequences))
    plot_indices = stratified_sampling(classes=kmed.labels_, sample_size=to_chose)

    y_true_all = []
    y_pred_all = []

    for i_seq in tqdm(range(len(sequences)), desc="Analysing test cases"):
        scaled = scaled_sequences[i_seq]

        window_ranges = [w for w in
                         windowed(np.arange(scaled.shape[0]), window_size=bd.window_size * 2,
                                  stride=max(1, int(STRIDE_RATE * bd.window_size)))]

        filtered_window_ranges = [wr for wr in window_ranges if len(wr) >= bd.window_size * (1 + MIN_X_WINDOW_RATE)]

        if len(filtered_window_ranges) == 0:
            continue

        x_windows = [scaled.iloc[wr[:bd.window_size], :] for wr in filtered_window_ranges]
        y_windows = [scaled.iloc[wr[bd.window_size:], :] for wr in filtered_window_ranges]

        x_window_features = extract_seq_features(x_windows, input_cols=PATTERN_CLUSTER_COLS)
        y_window_features = extract_seq_features(y_windows, input_cols=PATTERN_CLUSTER_COLS)

        x_classes = bd.kmed.predict(x_window_features)
        y_classes = bd.kmed.predict(y_window_features)

        y_classes_pred = model.predict(x_windows).tolist()

        y_true_all.extend(y_classes)
        y_pred_all.extend(y_classes_pred)

        if i_seq in plot_indices:
            plot_breathing(
                sequences[i_seq][PATTERN_CLUSTER_COLS],
                ranges=filtered_window_ranges,
                y_labels_true=y_classes,
                y_labels_pred=y_classes_pred,
                n_classes=model.n_classes,
                plot_index=i_seq
            )

    save_report_and_conf_m(y_true_all, y_pred_all, cm_title="Test confusion matrix")
