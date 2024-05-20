from src.breathing_patterns.data.dataset import BreathingDataset
from src.breathing_patterns.models.BreathingPatterModel import BreathingPatternModel
from src.config.seeds import set_seed
from src.session.utils.save_plots import base_dir
from src.tools.run_utils import get_run_path

DATASET_PATH = "../../bp_dataset_creation_runs/run_1/breathing_dataset.pkl"
RUN_PATH = "../../bp_train_runs"


def train():
    bd = BreathingDataset.read(DATASET_PATH)
    model = BreathingPatternModel()

    model.fit(bd, batch_size=128, n_epochs=3000, es_patience=50)
    model.dump(f"{base_dir()}/model.pkl")

if __name__ == '__main__':
    set_seed()
    run_path = get_run_path(RUN_PATH)
    print(f"Current run path: {run_path}")
    base_dir(run_path)

    train()

    pass
