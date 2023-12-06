import pandas as pd

from src.config.parsing import ModelParams, TrainParams
from src.models.state_prediction_module import StatePredictionModule


def train_model(
        model_params: ModelParams,
        train_params: TrainParams,
        sequences: list[pd.DataFrame]
):
    if train_params.sequence_limit is not None:
        sequences = sequences[:train_params.sequence_limit]

    model = StatePredictionModule(params=model_params, n_attr_in=sequences[0].shape[1])
    model.train(sequences=sequences, params=train_params)

    return model
