import pandas as pd

from src.models.window.window_model import WindowModel
from src.session.helpers.model_payload import SessionPayload


def train_windowed_model(
        payload: SessionPayload,
        sequences: list[pd.DataFrame]
):
    model = WindowModel(params=payload.model_params,
                        n_attr_in=sequences[0].shape[1],
                        window_size=12
                        )
    model.train(sequences=sequences, params=payload.train_params)

    return model
