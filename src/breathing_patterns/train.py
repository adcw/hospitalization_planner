from src.breathing_patterns.data.dataset import BreathingDataset
from src.breathing_patterns.models.BreathingPatterModel import BreathingPatternModel

DATASET_PATH = "../../bp_dataset_creation_runs/run_1/breathing_dataset.pkl"

if __name__ == '__main__':
    bd = BreathingDataset.read(DATASET_PATH)
    model = BreathingPatternModel()

    model.fit(bd, batch_size=16)

    pass
