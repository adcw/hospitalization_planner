from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.session.helpers.session_payload import SessionPayload
from src.visualization.predictions import PredictionData, plot_sequences_with_predictions


def test_model(
        session_payload: SessionPayload,
        sequences: list[pd.DataFrame],
        plot_indexes: Optional[list[int]] = None,
        limit: Optional[int] = None,

        max_per_sequence: Optional[int] = None,
        plot: bool = True,
        mode: str = "full"
):
    """

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

        if len(points) == 0:
            continue

        predictions: List[Tuple[int, np.iterable]] = []
        plot_this_seq = plot_indexes is None or seq_i in plot_indexes

        for point in points:
            cnt += 1

            x_real = seq[:point]

            y_real = seq[point:point + session_payload.main_params.n_steps_predict] \
                         .iloc[:, session_payload.model.target_col_indexes].values

            if y_real.shape[0] != session_payload.main_params.n_steps_predict or x_real.shape[0] == 0:
                continue

            y_real_raw = session_payload.model.data_transform(y_real)

            y_pred_raw = session_payload.model.predict(x_real, return_inv_transformed=False)
            y_pred = session_payload.model.data_inverse_transform(y_pred_raw)

            y_pred = y_pred.reshape((y_pred.shape[1], -1))

            # Calculate losses
            loss = np.mean(abs(np.array(y_real_raw) - np.array(y_pred_raw)))
            loss_sum += loss
            loss_calc_count += 1

            # Save plot data
            if plot and plot_this_seq:
                predictions.append((point, y_pred))

            if limit is not None and cnt >= limit:
                end = True
                break

        if plot and len(predictions) > 0:
            pred_data = PredictionData(target_col.values, predictions)
            plot_data.append(pred_data)

        if end:
            break

    if plot:
        plot_sequences_with_predictions(plot_data)

    return loss_sum / loss_calc_count if loss_calc_count != 0 else None
