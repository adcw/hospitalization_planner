from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.session.helpers.session_payload import SessionPayload
from src.visualization.predictions import PredictionData, plot_sequences_with_predictions


def test_model(
        session_payload: SessionPayload,
        sequences: list[pd.DataFrame],
        limit: Optional[int] = None,
        offset: int = 0,

        max_per_sequence: Optional[int] = None,
        plot: bool = True
):
    """

    :param session_payload:
    :param sequences:
    :param limit:
    :param offset:
    :param max_per_sequence:
    :param plot:
    :return: Average MAE loss
    """
    cnt = 0
    end = False

    plot_data: List[PredictionData] = []
    loss_sum = 0
    loss_calc_count = 0

    for seq in tqdm(sequences[offset:], desc="Analysing test cases"):

        target_col = seq[session_payload.main_params.cols_predict]

        if session_payload.test_params.mode == "full":
            points = [i for i in range(2, len(seq - session_payload.main_params.n_steps_predict))]
        elif session_payload.test_params.mode == "pessimistic":
            diff = np.diff(target_col, axis=0)
            points, _ = np.where(diff != 0)
            points += 1
        else:
            raise ValueError(f"Unsuported test mode: {session_payload.test_params.mode}")

        # If there is more points than max allowed, get random points of max count allowed.
        if max_per_sequence is not None and len(points) > max_per_sequence:
            points = np.random.choice(points, size=max_per_sequence, replace=False)

        if len(points) == 0:
            continue

        predictions: List[Tuple[int, np.iterable]] = []

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
