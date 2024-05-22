from sklearn.model_selection import StratifiedKFold

from src.breathing_patterns.data.generate_data import generate_breathing_dataset, extract_patterns, get_dataset
from src.breathing_patterns.models.BreathingPatterModel import BreathingPatternModel
from src.breathing_patterns.pattern_cluster_cols import PATTERN_CLUSTER_COLS
from src.breathing_patterns.perform_model_test import test_model
from src.breathing_patterns.utils.clustering import label_sequences
from src.chosen_colnames import COLS
from src.config.seeds import set_seed
from src.session.model_manager import _get_sequences
from src.session.utils.save_plots import base_dir
from src.tools.dataframe_scale import scale
from src.tools.reports import save_report_and_conf_m
from src.tools.run_utils import get_run_path

PATTERN_WINDOW_SIZE = 3
HISTORY_WINDOW_SIZE = 7

STRIDE_RATE = 0.001
N_CLASSES = 5
CSV_PATH = '../../data/input.csv'

RUN_PATH = "../../bp_crossvalidation_runs"

N_SPLITS = 10
EPOCHS = 5000

if __name__ == '__main__':
    set_seed()
    run_path = get_run_path(RUN_PATH)
    print(f"Current run path: {run_path}")

    sequences, preprocessor = _get_sequences(path=CSV_PATH, usecols=COLS)
    labels = label_sequences(sequences, stratify_cols=PATTERN_CLUSTER_COLS, n_clusters=4).labels_

    base_dir(f"{run_path}/pattern_extraction")
    kmed = extract_patterns(
        sequences=sequences,
        n_classes=N_CLASSES,
        stride_rate=STRIDE_RATE,
        pattern_window_size=PATTERN_WINDOW_SIZE,
        pattern_cluster_cols=PATTERN_CLUSTER_COLS
    )

    kfold = StratifiedKFold(n_splits=N_SPLITS)

    y_true_all = []
    y_pred_all = []
    for i, (train_indices, test_indices) in enumerate(kfold.split(sequences, y=labels)):
        print(f"SPLIT {i + 1}")
        sequences_train = [sequences[i] for i in train_indices]
        sequences_test = [sequences[i] for i in test_indices]

        base_dir(f"{run_path}/split_{i + 1}/dataset")
        bd = get_dataset(
            sequences_train=sequences_train,
            sequences_test=sequences_test,

            pattern_window_size=PATTERN_WINDOW_SIZE,
            history_window_size=HISTORY_WINDOW_SIZE,

            pattern_cluster_cols=PATTERN_CLUSTER_COLS,
            stride_rate=STRIDE_RATE,

            kmed=kmed
        )

        base_dir(f"{run_path}/split_{i + 1}/train")
        model = BreathingPatternModel(window_size=HISTORY_WINDOW_SIZE)
        model.fit(bd, batch_size=128, n_epochs=EPOCHS, es_patience=50)

        # Classification report in dict and confusion matrix
        base_dir(f"{run_path}/split_{i + 1}/test")
        y_true, y_pred = test_model(model=model, dataset=bd)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    base_dir(f"{run_path}")
    save_report_and_conf_m(y_true=y_true_all, y_pred=y_pred_all, cm_title="Macierz pomyłek - dane testowe")
    print("Crossvalidation done")