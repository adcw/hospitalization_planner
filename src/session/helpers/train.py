import pandas as pd

from src.models.stateful_prediction_module import StatePredictionModule
from src.session.helpers.model_payload import SessionPayload


def train_model(
        payload: SessionPayload,
        sequences: list[pd.DataFrame]
):
    model = StatePredictionModule(params=payload.model_params,
                                  n_attr_in=sequences[0].shape[1]
                                  )
    model.train(sequences=sequences, params=payload.train_params)

    return model
