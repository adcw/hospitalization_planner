from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models.step.forward import forward_sequences
from src.session.helpers.session_payload import SessionPayload
from src.visualization.predictions import PredictionData, plot_sequences_with_predictions


def test_model_state_optimal(
        session_payload: SessionPayload,
        sequences: list[pd.DataFrame],
        plot_indexes: Optional[list[int]] = None,

        plot: bool = True,
        max_plots: int = 12,
):
    """
    Perform full test on full sequence with stateful model
    :return:
    """

    assert session_payload.main_params.model_type == "step", \
        "test_model_state_optimal should be used only for stateful model"

    plot_data: List[PredictionData] = []

    seq_losses = []
    loss_calc_count = 0

    scaler = session_payload.model.scaler

    for seq_i, seq in tqdm(enumerate(sequences), desc="Analysing test cases"):
        target_col = seq[session_payload.main_params.cols_predict]
        target_col_indexes = [sequences[0].columns.values.tolist().index(col) for col in
                              session_payload.main_params.cols_predict] \
            if session_payload.main_params.cols_predict is not None else None

        plot_this_seq = plot_indexes is None or seq_i in plot_indexes

        seq_raw = scaler.transform(seq.values)
        seq_raw = torch.Tensor(seq_raw).to(session_payload.main_params.device)

        _, mae_loss, preds = forward_sequences([seq_raw], is_eval=True,
                                               model=session_payload.model.model,
                                               main_params=session_payload.main_params,
                                               optimizer=session_payload.model.optimizer,
                                               criterion=session_payload.model.criterion,
                                               target_indexes=target_col_indexes,
                                               y_cols_in_x=session_payload.main_params.cols_predict_training,
                                               verbose=False)

        seq_losses.append(float(mae_loss))
        loss_calc_count += 1

        predictions: List[Tuple[int, np.iterable]] = []
        if plot and plot_this_seq:
            # Save plot data
            for i, j in zip(range(1, len(seq)), preds):
                predictions.append((i, j))

            pred_data = PredictionData(target_col.values, predictions)
            plot_data.append(pred_data)

    if plot:
        plot_sequences_with_predictions(plot_data, max_plots=max_plots)

    return sum(seq_losses) / loss_calc_count if loss_calc_count != 0 else None, seq_losses


def test_model(
        session_payload: SessionPayload,
        sequences: list[pd.DataFrame],
        plot_indexes: Optional[list[int]] = None,
        limit: Optional[int] = None,

        max_per_sequence: Optional[int] = None,
        plot: bool = True,
        mode: str = "full",
        max_plots: int = 12,
):
    """
    Perform naive testing

    :param session_payload:
    :param sequences:
    :param limit:
    :param max_per_sequence:
    :param plot:
    :return: Average MAE loss
    """
    cnt = 0
    end = False

    plot_data: List[PredictionData] = []
    loss_sum = 0
    loss_calc_count = 0

    y_columns = [sequences[0].columns.get_loc(col) for col in session_payload.main_params.cols_predict]

    x_cols = set(range(0, sequences[0].shape[1]))

    if not session_payload.main_params.cols_predict_training:
        x_cols = x_cols.difference(y_columns)

    x_cols = list(x_cols)
    seq_losses = []

    for seq_i, seq in tqdm(enumerate(sequences), desc="Analysing test cases"):
        target_col = seq[session_payload.main_params.cols_predict]

        if mode == "full":
            points = [i for i in range(2, len(seq - session_payload.main_params.n_steps_predict))]
        elif mode == "pessimistic":
            diff = np.diff(target_col, axis=0)
            points, _ = np.where(diff != 0)
            points += 1
        else:
            raise ValueError(f"Unsuported test mode: {mode}")

        # If there is more points than max allowed, get random points of max count allowed.
        if max_per_sequence is not None and len(points) > max_per_sequence:
            points = np.random.choice(points, size=max_per_sequence, replace=False)

        predictions: List[Tuple[int, np.iterable]] = []
        plot_this_seq = plot_indexes is None or seq_i in plot_indexes

        seq_loss = 0

        for point in points:
            cnt += 1

            x_real = seq.iloc[:point, x_cols]

            y_real = seq[point:point + session_payload.main_params.n_steps_predict] \
                         .iloc[:, session_payload.model.target_col_indexes].values

            if y_real.shape[0] != session_payload.main_params.n_steps_predict or x_real.shape[0] == 0:
                continue

            y_real_raw = session_payload.model.transform_y(y_real)
            y_pred_raw = session_payload.model.predict(x_real, return_inv_transformed=False)
            y_pred = session_payload.model.inverse_transform_y(y_pred_raw)

            y_pred = y_pred.reshape((y_pred.shape[1], -1))

            # Calculate losses
            loss = np.mean(abs(np.array(y_real_raw) - np.array(y_pred_raw)))
            loss_sum += loss
            seq_loss += loss
            loss_calc_count += 1

            # Save plot data
            if plot and plot_this_seq:
                predictions.append((point, y_pred))

            if limit is not None and cnt >= limit:
                end = True
                break

        seq_losses.append(seq_loss / len(points) if len(points) > 0 else None)

        if plot and plot_this_seq > 0:
            pred_data = PredictionData(target_col.values, predictions)
            plot_data.append(pred_data)

        if end:
            break

    if plot:
        plot_sequences_with_predictions(plot_data, max_plots=max_plots)

    return loss_sum / loss_calc_count if loss_calc_count != 0 else None, seq_losses
