import pandas as pd

from src.models.step.step_model import StepModel
from src.models.window.window_model import WindowModel
from src.session.helpers.session_payload import SessionPayload


def train_model(
        payload: SessionPayload,
        sequences: list[pd.DataFrame]
):
    # model = StepModel(params=payload.model_params,
    #                   n_attr_in=sequences[0].shape[1]
    #                   )
    # model.train(sequences=sequences, params=payload.train_params)

    model = WindowModel(main_params=payload.main_params,
                        n_attr_in=sequences[0].shape[1],
                        window_size=12
                        )
    train_mae_losses, val_mae_losses = model.train(sequences=sequences, params=payload.train_params)

    return model, (train_mae_losses, val_mae_losses)
