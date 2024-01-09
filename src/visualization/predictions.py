from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_pred_comparison(plot_data: List[Tuple[np.array, np.array]]):
    num_subplots = len(plot_data)

    num_cols = int(np.ceil(np.sqrt(num_subplots)))
    num_rows = int(np.ceil(num_subplots / num_cols))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))

    if num_subplots == 1:
        y_real, y_pred = plot_data[0]
        axs.plot(y_real, label="real")
        axs.plot(y_pred, label="predicted")
    else:
        for i in range(num_rows):
            if len(plot_data) == 0:
                break

            for j in range(num_cols):
                if len(plot_data) == 0:
                    break

                if num_subplots > 0:
                    y_real, y_pred = plot_data.pop(0)
                    axs[i, j].plot(y_real, label=f'real')
                    axs[i, j].plot(y_pred, label=f'pred')
                    axs[i, j].legend()
                    axs[i, j].set_ylim(-0.1, 1.1)
                else:
                    axs[i, j].axis('off')
                    axs[i, j].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.suptitle("Predictions")
    plt.show()


@dataclass
class PredictionData:
    original_sequence: np.ndarray
    preds: List[Tuple[int, np.iterable]]


def plot_sequences_with_predictions(
        data: List[PredictionData]
):
    max_plots = 12
    num_entries = len(data)

    cols = 2

    num_plots = min(max_plots, num_entries)
    num_rows = num_plots // cols if num_plots % cols == 0 else (num_plots // cols) + 1

    fig, axes = plt.subplots(num_rows, cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i in range(num_plots):
        entry = data[i]
        original_sequence = entry.original_sequence
        preds = entry.preds

        ax = axes[i]

        ax.plot(original_sequence, "--", label='Original Sequence')

        for index, pred_sequence in preds:
            pred_start = max(index, 0)
            pred_end = pred_start + len(pred_sequence)
            ax.plot(range(pred_start, pred_end), pred_sequence, ".",
                    label=f'Prediction at Index {index}')

        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title(f'Entry {i + 1}')

    plt.tight_layout()

    pass


if __name__ == '__main__':
    # test plot_sequence_with_predictions
    data = np.array([1, 2, 3, 3, 4, 5, 4, 3, 2])
    preds: List[Tuple[int, np.ndarray]] = [
        (1, np.array([5, 6, 7])),
        (5, np.array([5, 5, 4])),
    ]

    data = [PredictionData(data, preds)]

    plot_sequences_with_predictions(data)
    plt.show()

    pass
