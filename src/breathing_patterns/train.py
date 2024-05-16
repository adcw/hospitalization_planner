from src.breathing_patterns.data.dataset import BreathingDataset
from src.breathing_patterns.models.BreathingPatterModel import BreathingPatternModel
from src.config.seeds import set_seed
from src.session.utils.save_plots import base_dir
from src.tools.run_utils import get_run_path

DATASET_PATH = "../../bp_dataset_creation_runs/run_2/breathing_dataset.pkl"
RUN_PATH = "../../bp_train_runs"

if __name__ == '__main__':
    set_seed()
    run_path = get_run_path(RUN_PATH)
    print(f"Current run path: {run_path}")
    base_dir(run_path)

    bd = BreathingDataset.read(DATASET_PATH)
    model = BreathingPatternModel()

    model.fit(bd, batch_size=16, n_epochs=400, es_patience=20)
    model.dump(f"{base_dir()}/model.pkl")

    pass
