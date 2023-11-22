from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.parsing import ModelParams, TrainParams
from src.nn import StatePredictionModule
from src.plotting import plot_trajectories, plot_pred_comparison


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

        max_per_sequence: int = 3
):
    cnt = 0
    end = False

    plot_data = []

    for seq in tqdm(sequences[offset:], desc="Evaluating test cases"):
        target_col = seq[model_payload.model_params.cols_predict]
        diff = np.diff(target_col, axis=0)
        points, _ = np.where(diff != 0)

        # If there is more points than max allowed, get random points of max count allowed.
        if len(points) > 3:
            points = np.random.choice(points, size=max_per_sequence, replace=False)

        for point in points:
            cnt += 1

            x_real = seq[:point]

            if x_real.shape[0] == 0:
                continue

            y_real = seq[point:point + model_payload.model_params.n_steps_predict] \
                         .iloc[:, model_payload.model.target_col_indexes].values

            y_pred = model_payload.model.predict(x_real)
            y_pred = y_pred.reshape((y_pred.shape[1], -1))

            plot_data.append((y_real, y_pred))

            if cnt >= limit:
                end = True
                break

        if end:
            break

    plot_pred_comparison(plot_data)


def load_data_helper():
    pass
