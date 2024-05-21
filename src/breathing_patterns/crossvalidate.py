from sklearn.model_selection import StratifiedKFold

from src.breathing_patterns.data.generate_data import generate_breathing_dataset
from src.breathing_patterns.models.BreathingPatterModel import BreathingPatternModel
from src.breathing_patterns.pattern_cluster_cols import PATTERN_CLUSTER_COLS
from src.breathing_patterns.perform_model_test import test_model
from src.breathing_patterns.utils.clustering import label_sequences
from src.chosen_colnames import COLS
from src.config.seeds import set_seed
from src.session.model_manager import _get_sequences
from src.session.utils.save_plots import base_dir
from src.tools.reports import save_report_and_conf_m
from src.tools.run_utils import get_run_path

PATTERN_WINDOW_SIZE = 5
HISTORY_WINDOW_SIZE = 15

STRIDE_RATE = 0.001
N_CLASSES = 5
TEST_PERC = 0.2
CSV_PATH = '../../data/input.csv'

RUN_PATH = "../../bp_crossvalidation_runs"

if __name__ == '__main__':
    set_seed()
    run_path = get_run_path(RUN_PATH)
    print(f"Current run path: {run_path}")

    sequences, preprocessor = _get_sequences(path=CSV_PATH, usecols=COLS)
    labels = label_sequences(sequences, stratify_cols=PATTERN_CLUSTER_COLS).labels_

    kfold = StratifiedKFold(n_splits=10)

    y_true_all = []
    y_pred_all = []
    for i, (train_indices, test_indices) in enumerate(kfold.split(sequences, y=labels)):
        print(f"SPLIT {i + 1}")
        sequences_train = [sequences[i] for i in train_indices]
        sequences_test = [sequences[i] for i in test_indices]

        base_dir(f"{run_path}/split_{i + 1}/dataset")
        bd = generate_breathing_dataset(
            sequences_train=sequences_train,
            sequences_test=sequences_test,

            pattern_window_size=PATTERN_WINDOW_SIZE,
            history_window_size=HISTORY_WINDOW_SIZE,

            stride_rate=STRIDE_RATE,
            n_classes=N_CLASSES,
            pattern_cluster_cols=PATTERN_CLUSTER_COLS,
        )

        base_dir(f"{run_path}/split_{i + 1}/train")
        model = BreathingPatternModel()
        model.fit(bd, batch_size=128, n_epochs=5000, es_patience=50)

        # Classification report in dict and confusion matrix
        base_dir(f"{run_path}/split_{i + 1}/test")
        y_true, y_pred = test_model(model=model, dataset=bd)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    base_dir(f"{run_path}")
    save_report_and_conf_m(y_true=y_true_all, y_pred=y_pred_all, cm_title="Macierz pomyłek - dane testowe")
