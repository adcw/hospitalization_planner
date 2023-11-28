from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.config.parsing import ModelParams, TrainParams, EvalParams
from src.nn import StatePredictionModule
from src.plotting import PredictionData, plot_sequences_with_predictions


@dataclass
class ModelPayload:
    model_params: ModelParams
    train_params: TrainParams
    eval_params: EvalParams
    model: Optional[StatePredictionModule]


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


def eval_model_helper(
        payload: ModelPayload,
        sequences: list[pd.DataFrame],
):
    # Retrieve params, prepare training params to perform CV
    eval_params = payload.eval_params
    train_params = deepcopy(payload.train_params)
    train_params.epochs = eval_params.epochs
    train_params.es_patience = eval_params.es_patience

    kf = KFold(n_splits=eval_params.n_splits, shuffle=True)

    val_losses = []
    train_losses = []

    for split_i, (train_index, val_index) in enumerate(kf.split(sequences)):
        print(f"Training on split number {split_i + 1}")

        model = StatePredictionModule(payload.model_params, n_attr_in=sequences[0].shape[1])

        # Get train and validation tensors
        train_sequences = [sequences[i] for i in train_index]
        val_sequences = [sequences[i] for i in val_index]

        # Train on sequences
        train_loss = model.train(train_params, train_sequences, plot=False)
        model_payload = deepcopy(payload)
        model_payload.model = model

        # Perform test
        mean_val_loss = test_model_helper(model_payload, val_sequences, limit=None, plot=False, max_per_sequence=None)

        # TODO: Mak eot cleaner
        val_losses.append(mean_val_loss)
        train_losses.append(train_loss)
        print(f"Mean validation loss: {mean_val_loss}")

    plt.plot(val_losses, 'o', label="val_loss")
    plt.plot(train_losses, 'o', label="train_loss")

    # Adding text labels near data markers
    for i, value in enumerate(val_losses):
        s = f'{value:.4f}' if value is not None else ""
        plt.text(i, value, s, ha='center', va='bottom')

    for i, value in enumerate(train_losses):
        plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')

    plt.title(f"Losses on each fold. Avg = {np.average(val_losses)}")
    plt.legend()
    plt.grid(True)
    plt.show()

    pass


def test_model_helper(
        model_payload: ModelPayload,
        sequences: list[pd.DataFrame],
        limit: Optional[int] = 30,
        offset: int = 0,

        max_per_sequence: Optional[int] = 3,
        plot: bool = True
):
    cnt = 0
    end = False

    plot_data: List[PredictionData] = []
    loss_sum = 0
    loss_calc_count = 0

    for seq in tqdm(sequences[offset:], desc="Evaluating test cases"):
        target_col = seq[model_payload.model_params.cols_predict]
        diff = np.diff(target_col, axis=0)
        points, _ = np.where(diff != 0)

        # If there is more points than max allowed, get random points of max count allowed.
        if max_per_sequence is not None and len(points) > max_per_sequence:
            points = np.random.choice(points, size=max_per_sequence, replace=False)

        if len(points) == 0:
            continue

        predictions: List[Tuple[int, np.iterable]] = []

        for point in points:
            cnt += 1

            x_real = seq[:point]

            if x_real.shape[0] != model_payload.model_params.n_steps_predict:
                continue

            y_real = seq[point:point + model_payload.model_params.n_steps_predict] \
                         .iloc[:, model_payload.model.target_col_indexes].values

            y_pred = model_payload.model.predict(x_real)
            y_pred = y_pred.reshape((y_pred.shape[1], -1))

            # Calculate losses
            loss = np.mean((y_real - np.array(y_pred)) ** 2)
            loss_sum += loss
            loss_calc_count += 1

            # Save plot data
            predictions.append((point, y_pred))

            if limit is not None and cnt >= limit:
                end = True
                break

        pred_data = PredictionData(target_col.values, predictions)

        if plot:
            plot_data.append(pred_data)

        if end:
            break

    if plot:
        plot_sequences_with_predictions(plot_data)

    return loss_sum / loss_calc_count if loss_calc_count != 0 else None
