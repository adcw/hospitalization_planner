import numpy as np

from src.breathing_patterns.data.dataset import BreathingDataset
from src.breathing_patterns.models.BreathingPatterModel import BreathingPatternModel
from src.breathing_patterns.utils.clustering import learn_clusters
from src.model_selection.stratified_sampling import stratified_sampling
from src.session.utils.save_plots import base_dir
from src.tools.dataframe_scale import scale
from src.tools.iterators import windowed
from src.tools.run_utils import get_run_path

RESPIRATION = 'respiration'
GENERAL_SURFACTANT = 'GENERAL_SURFACTANT'
FIO2 = 'FiO2'
RTG_RDS = 'RTG_RDS'
PO2 = 'po2'

RESP_COLS = [
    RESPIRATION, GENERAL_SURFACTANT, FIO2, RTG_RDS, PO2
]

DATASET_PATH = "../../bp_dataset_creation_runs/run_1/breathing_dataset.pkl"
RUN_PATH = "../../bp_test_runs"
MODEL_PATH = "../../bp_train/run_3/model.pkl"

MAX_PLOTS = 5
MIN_X_WINDOW_RATE = 1
STRIDE_RATE = 0.5

if __name__ == '__main__':
    run_path = get_run_path(RUN_PATH)
    base_dir(run_path)

    bd = BreathingDataset.read(DATASET_PATH)
    model = BreathingPatternModel()
    model.load(MODEL_PATH)

    sequences = bd.test_sequences
    scaled_sequences, scaler = scale(sequences)
    kmed = learn_clusters(scaled_sequences, n_clusters=7, input_cols=[RESPIRATION], save_plots=False)

    # Perform quick test
    to_chose = min(MAX_PLOTS, len(bd.test_sequences))
    plot_indices = stratified_sampling(classes=kmed.labels_, sample_size=to_chose)

    for i_seq in range(len(sequences)):
        scaled = scaled_sequences[i_seq]

        window_ranges = [w for w in
                         windowed(np.arange(scaled.shape[0]), window_size=bd.window_size * 2,
                                  stride=int(STRIDE_RATE * bd.window_size))]

        filtered_window_ranges = [wr for wr in window_ranges if len(wr) >= bd.window_size * (1 + MIN_X_WINDOW_RATE)]

        if len(filtered_window_ranges) == 0:
            continue

        x_windows = [scaled.iloc[wr[:bd.window_size], :] for wr in filtered_window_ranges]
        y_windows = [scaled.iloc[wr[bd.window_size:], :] for wr in filtered_window_ranges]

        pass

    pass
