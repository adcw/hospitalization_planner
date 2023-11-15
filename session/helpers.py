from dataclasses import dataclass

from models.utils import get_sequences
from src.config.config_classes import ModelParams, TrainParams
from src.nn import StatePredictionModule


@dataclass
class ModelPayload:
    model: StatePredictionModule
    model_params: ModelParams
    train_params: TrainParams


def train_model_helper(
        model_params: ModelParams,
        train_params: TrainParams
):
    # TODO: get sequences nicer
    sequences, preprocessor = get_sequences()

    # TODO: delete this hardcoded
    sequences = sequences[:3]
    train_params.epochs = 4

    model = StatePredictionModule(params=model_params, n_attr_in=sequences[0].shape[1])
    model.train(sequences=sequences, params=train_params)

    return model


def test_model_helper(
        model_payload: ModelPayload
):
    print(f"The model will be tested.")
    pass
