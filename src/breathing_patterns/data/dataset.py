import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sklearn_extra.cluster


class BreathingDataset:
    def __init__(self,
                 xs: List[pd.DataFrame],
                 xs_classes: np.ndarray,

                 ys_classes: np.ndarray,

                 test_sequences: List[pd.DataFrame],

                 pattern_window_size: int,
                 history_window_size: int,

                 kmed: sklearn_extra.cluster.KMedoids,
                 scaler: MinMaxScaler
                 ):
        self.pattern_window_size = pattern_window_size
        self.history_window_size = history_window_size

        self.xs = xs
        self.xs_classes = xs_classes
        self.scaler = scaler

        self.ys_classes = ys_classes

        self.test_sequences = test_sequences

        self.kmed = kmed

        self.cache_path: Optional[str] = None

    @staticmethod
    def read(path: str):
        with open(path, "rb") as file:
            xs, xs_classes, ys_classes, test_sequences, hws, pws, kmed, scaler = pickle.load(file)
            return BreathingDataset(
                xs=xs,
                xs_classes=xs_classes,

                ys_classes=ys_classes,
                test_sequences=test_sequences,

                history_window_size=hws,
                pattern_window_size=pws,

                kmed=kmed,
                scaler=scaler
            )

    def save(self, path: str):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb+") as file:
            pickle.dump((
                self.xs,
                self.xs_classes,

                self.ys_classes,
                self.test_sequences,

                self.history_window_size,
                self.pattern_window_size,

                self.kmed,
                self.scaler
            ), file)

        self.cache_path = path
