from src.breathing_patterns.data.dataset import BreathingDataset
from src.breathing_patterns.models.BreathingPatterModel import BreathingPatternModel
from src.session.utils.save_plots import base_dir
from src.tools.run_utils import get_run_path

DATASET_PATH = "../../bp_dataset_creation_runs/run_3/breathing_dataset.pkl"
RUN_PATH = "../../bp_train_runs"

if __name__ == '__main__':
    run_path = get_run_path(RUN_PATH)
    base_dir(run_path)

    bd = BreathingDataset.read(DATASET_PATH)
    model = BreathingPatternModel()

    model.fit(bd, batch_size=16, n_epochs=100)
    model.dump(f"{base_dir()}/model.pkl")

    pass
