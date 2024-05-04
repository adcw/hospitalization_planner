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

                 window_size: int,
                 kmed: sklearn_extra.cluster.KMedoids,
                 scaler: MinMaxScaler
                 ):
        self.window_size = window_size

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
            xs, xs_classes, ys_classes, test_sequences, kmed, scaler = pickle.load(file)
            return BreathingDataset(
                xs=xs,
                xs_classes=xs_classes,

                ys_classes=ys_classes,
                test_sequences=test_sequences,

                window_size=xs[0].shape[0],
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

                self.kmed,
                self.scaler
            ), file)

        self.cache_path = path
