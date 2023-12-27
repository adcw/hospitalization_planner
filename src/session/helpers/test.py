from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.session.helpers.model_payload import SessionPayload
from src.visualization.predictions import PredictionData, plot_sequences_with_predictions


def test_model(
        model_payload: SessionPayload,
        sequences: list[pd.DataFrame],
        limit: Optional[int] = 30,
        offset: int = 0,

        max_per_sequence: Optional[int] = 10,
        plot: bool = True
):
    cnt = 0
    end = False

    plot_data: List[PredictionData] = []
    loss_sum = 0
    loss_calc_count = 0

    for seq in tqdm(sequences[offset:], desc="Analysing test cases"):
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

            y_real = seq[point:point + model_payload.model_params.n_steps_predict] \
                         .iloc[:, model_payload.model.target_col_indexes].values

            if y_real.shape[0] != model_payload.model_params.n_steps_predict or x_real.shape[0] == 0:
                continue

            y_real_raw = model_payload.model.data_transform(y_real)

            y_pred_raw = model_payload.model.predict(x_real, return_inv_transformed=False)
            y_pred = model_payload.model.data_inverse_transform(y_pred_raw)

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
