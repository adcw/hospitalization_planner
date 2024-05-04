from typing import List, Optional
import pickle

import numpy as np
import pandas as pd

import os

import torch


class BreathingDataset:
    def __init__(self,
                 xs: List[pd.DataFrame],
                 ys: np.ndarray,
                 window_size: int
                 ):
        self.window_size = window_size

        self.xs = xs
        self.ys = ys

        self.cache_path: Optional[str] = None

    @staticmethod
    def read(path: str):
        with open(path, "rb") as file:
            xs, ys = pickle.load(file)
            return BreathingDataset(xs, ys, xs[0].shape[0])

    def save(self, path: str):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb+") as file:
            pickle.dump((self.xs, self.ys), file)

        self.cache_path = path
