from models.utils import get_sequences
from src.config.config_classes import TrainParams, ModelParams
from src.nn import StatePredictionModule

CONFIG_FILE_PATH = "./config.yaml"


def main():
    # TODO: Get rid of these
    # sequences, preprocessor = get_sequences()
    # sequences = sequences[:10]
    #
    # train_seq = sequences[:-1]
    # test_seq = sequences[-1]
    #
    # pred_model = StatePredictionModule(params=ModelParams.from_yaml(CONFIG_FILE_PATH), n_attr_in=sequences[0].shape[1])
    # pred_model.train(sequences=train_seq, params=TrainParams.from_yaml(CONFIG_FILE_PATH))
    #
    # trim_place = round(0.5 * len(test_seq))
    # prediction, _ = pred_model.predict(test_seq[:trim_place])

    pass


if __name__ == '__main__':
    main()
