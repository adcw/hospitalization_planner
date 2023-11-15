from dataclasses import dataclass

import matplotlib.pyplot as plt

from models.utils import get_sequences
from src.config.config_classes import ModelParams, TrainParams
from src.nn import StatePredictionModule

import pandas as pd

from src.preprocessing import Preprocessor


@dataclass
class ModelPayload:
    model: StatePredictionModule
    model_params: ModelParams
    train_params: TrainParams


def train_model_helper(
        model_params: ModelParams,
        train_params: TrainParams,
        sequences: list[pd.DataFrame]
):
    # TODO: delete this hardcoded
    sequences = sequences[:20]
    train_params.epochs = 30

    model = StatePredictionModule(params=model_params, n_attr_in=sequences[0].shape[1])
    model.train(sequences=sequences, params=train_params)

    return model


def test_model_helper(
        model_payload: ModelPayload,
        sequences: list[pd.DataFrame],
        preprocessor: Preprocessor
):
    # Take one for now
    sequence = sequences[3]
    split_point = round(0.5 * len(sequence))
    input_sequence = sequence[:split_point]

    pred = model_payload.model.predict(input_sequence)
    pred_inv = preprocessor.inverse_transform(pred, col_indexes=model_payload.model.target_col_indexes)

    real_trajectory = sequence[split_point:split_point + model_payload.model_params.n_steps_predict].iloc[:,
                      model_payload.model.target_col_indexes].values

    pred = pred.reshape((pred.shape[1], -1))
    real = sequence[split_point:split_point + model_payload.model_params.n_steps_predict].iloc[
           :, model_payload.model.target_col_indexes].values

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(real, label="Real trajectory")
    axs[0].plot(pred, label="Predicted trajectory")
    axs[0].set_ylim(-0.1, 1.1)
    axs[0].legend()

    axs[1].plot(real, label="Real trajectory")
    axs[1].plot(pred, label="Predicted trajectory")
    axs[1].legend()

    plt.show()

    pass


def load_data_helper():
    pass
