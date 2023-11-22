from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.parsing import ModelParams, TrainParams
from src.nn import StatePredictionModule


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
    if train_params.sequence_limit is not None:
        sequences = sequences[:train_params.sequence_limit]

    model = StatePredictionModule(params=model_params, n_attr_in=sequences[0].shape[1])
    model.train(sequences=sequences, params=train_params)

    return model


def plot_trajectories(ts: List[np.iterable], title: str = "Trajectories"):
    num_subplots = min(len(ts), 6 * 6)
    num_cols = int(np.ceil(np.sqrt(num_subplots)))
    num_rows = int(np.ceil(num_subplots / num_cols))

    ts = ts.copy()

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_rows, 3 * num_cols))

    if len(ts) == 1:
        axs.plot(ts[0])
    else:
        for i in range(num_rows):
            for j in range(num_cols):
                if len(ts) > 0:
                    data = ts.pop(0)
                    axs[i, j].plot(data, label=f'Series {i * num_cols + j + 1}')
                    axs[i, j].legend()
                else:
                    axs[i, j].axis('off')

    plt.tight_layout()
    plt.suptitle(title)
    plt.show()


def plot_critical_places(seq: np.ndarray):
    diff = np.diff(seq.ravel())
    points = np.where(diff != 0)

    trajectories = []

    for point in points:
        start = point[0] - 30
        end = point[0] + 30

        trajectories.append(seq[start: end])

    plot_trajectories(trajectories)


def test_model_helper(
        model_payload: ModelPayload,
        sequences: list[pd.DataFrame],
        limit: int = 15,
        offset: int = 0,
):
    pass

    plot_critical_places(sequences[2][model_payload.model_params.cols_predict].values)

    for seq in tqdm(sequences[offset:offset + limit], desc="Evaluating test cases"):
        split_point = round(0.5 * len(seq))
        input_sequence: pd.DataFrame = seq[:split_point]

        pred = model_payload.model.predict(input_sequence)

        pred = pred.reshape((pred.shape[1], -1))

        real = seq[split_point:split_point + model_payload.model_params.n_steps_predict].iloc[
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
