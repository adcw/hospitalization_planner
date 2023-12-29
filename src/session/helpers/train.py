import pandas as pd

from src.models.step.step_model import StepModel
from src.session.helpers.model_payload import SessionPayload


def train_model(
        payload: SessionPayload,
        sequences: list[pd.DataFrame]
):
    model = StepModel(params=payload.model_params,
                      n_attr_in=sequences[0].shape[1]
                      )
    model.train(sequences=sequences, params=payload.train_params)

    return model
