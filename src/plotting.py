from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


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
                else:
                    axs[i, j].axis('off')

    plt.tight_layout()
    plt.suptitle("Predictions")
    plt.show()
