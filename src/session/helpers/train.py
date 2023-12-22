import pandas as pd

from src.models.state_prediction_module import StatePredictionModule
from src.session.helpers.model_payload import SessionPayload


def train_model(
        payload: SessionPayload,
        sequences: list[pd.DataFrame]
):
    # if payload.train_params.sequence_limit is not None:
    #     sequences = sequences[:payload.train_params.sequence_limit]

    model = StatePredictionModule(params=payload.model_params, n_attr_in=sequences[0].shape[1])
    model.train(sequences=sequences, params=payload.train_params)

    return model
