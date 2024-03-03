from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(ts: List[np.ndarray], title: str = "Trajectories"):
    num_subplots = min(len(ts), 6 * 6)
    num_cols = int(np.ceil(np.sqrt(num_subplots)))
    num_rows = int(np.ceil(num_subplots / num_cols))

    ts = ts.copy()

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))

    if len(ts) == 1:
        axs.plot(ts[0])
    else:
        for i in range(num_rows):
            for j in range(num_cols):
                if len(ts) > 0:
                    data = ts.pop(0)
                    if num_rows == 1 or num_cols == 1:
                        axs[max(i, j)].plot(data, label=f'Series {i * num_cols + j + 1}')
                        axs[max(i, j)].legend()
                    else:
                        axs[i, j].plot(data, label=f'Series {i * num_cols + j + 1}')
                        axs[i, j].legend()
                else:
                    axs[i, j].axis('off')

    plt.tight_layout()
    plt.suptitle(title)
    plt.show()
