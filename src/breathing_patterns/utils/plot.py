import numpy as np
import matplotlib.pyplot as plt

from src.session.utils.save_plots import save_plot


def plot_medoid_data(plot_data, pattern_cluster_cols):
    for medoid_data in plot_data:
        class_label = medoid_data['class']
        window_data = medoid_data['window']

        num_plots = len(pattern_cluster_cols)
        n_rows = int(np.sqrt(num_plots))
        n_cols = int(np.ceil(num_plots / n_rows))

        fig, axs = plt.subplots(n_cols, n_rows, figsize=(8, 10))
        axs = axs.flatten()

        for i, col in enumerate(pattern_cluster_cols):
            col_values = window_data[col].values

            axs[i].plot(col_values)
            axs[i].set_title(col)
            axs[i].set_xlabel('Time index')
            axs[i].set_ylabel('Value')

        # Dostosowanie układu subplotu
        plt.tight_layout()
        plt.suptitle(f"Centroid of class {class_label}")

        # Wyświetlanie wykresów
        save_plot(f"cluster_{class_label}")
        plt.show()
