import pandas as pd

from src.models.windowed_module import WindowedModule
from src.session.helpers.model_payload import SessionPayload


def train_windowed_model(
        payload: SessionPayload,
        sequences: list[pd.DataFrame]
):
    model = WindowedModule(params=payload.model_params,
                           n_attr_in=sequences[0].shape[1],
                           window_size=12
                           )
    model.train(sequences=sequences, params=payload.train_params)

    return model