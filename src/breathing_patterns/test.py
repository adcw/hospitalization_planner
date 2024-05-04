from src.breathing_patterns.data.dataset import BreathingDataset
from src.breathing_patterns.models.BreathingPatterModel import BreathingPatternModel
from src.session.utils.save_plots import base_dir
from src.tools.run_utils import get_run_path

DATASET_PATH = "../../bp_dataset_creation_runs/run_1/breathing_dataset.pkl"
RUN_PATH = "../../bp_test_runs"
MODEL_PATH = "../../bp_train/run_3/model.pkl"

if __name__ == '__main__':
    run_path = get_run_path(RUN_PATH)
    base_dir(run_path)

    bd = BreathingDataset.read(DATASET_PATH)
    model = BreathingPatternModel()
    model.load(MODEL_PATH)

    pass
