import pandas as pd

from src.models.step.step_model import StepModel
from src.models.window.window_model import WindowModel
from src.session.helpers.session_payload import SessionPayload


def train_model(
        payload: SessionPayload,
        sequences: list[pd.DataFrame]
):
    if payload.main_params.model_type == "step":

        model = StepModel(main_params=payload.main_params,
                          n_attr_in=sequences[0].shape[1]
                          )
        train_mae_losses, val_mae_losses, final_train_mae_loss, final_val_mae_loss = model.train(sequences=sequences,
                                                                                                 params=payload.train_params)
    else:
        model = WindowModel(main_params=payload.main_params,
                            n_attr_in=sequences[0].shape[1],
                            window_size=12
                            )
        train_mae_losses, val_mae_losses, final_train_mae_loss, final_val_mae_loss = model.train(sequences=sequences,
                                                                                                 params=payload.train_params)

    return model, (train_mae_losses, val_mae_losses, final_train_mae_loss, final_val_mae_loss)
