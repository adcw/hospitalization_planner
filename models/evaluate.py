from models.utils import get_sequences
from src.config.config_classes import TrainParams, ModelParams
from src.nn import StatePredictionModule

CONFIG_FILE_PATH = "./config.yaml"


def main():
    sequences, preprocessor = get_sequences()

    pred_model = StatePredictionModule(params=ModelParams.from_yaml(CONFIG_FILE_PATH), n_attr_in=sequences[0].shape[1])

    pred_model.eval(sequences=sequences[:10], params=TrainParams.from_yaml(CONFIG_FILE_PATH))

    pass


if __name__ == '__main__':
    main()
