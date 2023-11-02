from models.utils import get_sequences
from src.config.net_params import TrainParams
from src.nn import StatePredictionModule, NetParams

CONFIG_FILE_PATH = "./config.yaml"


def main():
    sequences, preprocessor = get_sequences()

    params = NetParams.from_yaml(yaml_path="./config.yaml")
    params.n_attr = sequences[0].shape[1]
    pred_model = StatePredictionModule(net_params=params)

    pred_model.train(sequences=sequences[:10], mode='eval', params=TrainParams.from_yaml(CONFIG_FILE_PATH))

    pass


if __name__ == '__main__':
    main()
