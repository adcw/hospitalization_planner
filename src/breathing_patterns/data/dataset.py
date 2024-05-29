import os
import pickle
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids


class BreathingDataset:
    def __init__(self,
                 xs: List[pd.DataFrame],
                 xs_classes: np.ndarray,
                 ys_classes: np.ndarray,
                 n_classes: int,
                 test_sequences: List[pd.DataFrame],
                 pattern_window_size: int,
                 history_window_size: int,
                 kmed: KMedoids,
                 scaler: MinMaxScaler,
                 cache_path: Optional[str] = None):
        """
        Initializes the BreathingDataset object.

        Args:
            xs (List[pd.DataFrame]): List of input dataframes.
            xs_classes (np.ndarray): Array of classes corresponding to xs.
            ys_classes (np.ndarray): Array of classes for output.
            n_classes (int): Number of classes.
            test_sequences (List[pd.DataFrame]): List of test sequences.
            pattern_window_size (int): Size of the pattern window.
            history_window_size (int): Size of the history window.
            kmed (KMedoids): KMedoids object for clustering.
            scaler (MinMaxScaler): Scaler for normalizing data.
            cache_path (Optional[str]): Optional path for caching the dataset.
        """
        self.pattern_window_size = pattern_window_size
        self.history_window_size = history_window_size
        self.xs = xs
        self.xs_classes = xs_classes
        self.ys_classes = ys_classes
        self.n_classes = n_classes
        self.test_sequences = test_sequences
        self.kmed = kmed
        self.scaler = scaler
        self.cache_path = cache_path

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the BreathingDataset object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the object.
        """
        return {
            'pattern_window_size': self.pattern_window_size,
            'history_window_size': self.history_window_size,
            'xs': self.xs,
            'xs_classes': self.xs_classes,
            'ys_classes': self.ys_classes,
            'n_classes': self.n_classes,
            'test_sequences': self.test_sequences,
            'kmed': self.kmed,
            'scaler': self.scaler,
            'cache_path': self.cache_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BreathingDataset':
        """
        Creates a BreathingDataset object from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary representation of the object.

        Returns:
            BreathingDataset: The created object.
        """
        return cls(
            xs=data['xs'],
            xs_classes=data['xs_classes'],
            ys_classes=data['ys_classes'],
            n_classes=data['n_classes'],
            test_sequences=data['test_sequences'],
            pattern_window_size=data['pattern_window_size'],
            history_window_size=data['history_window_size'],
            kmed=data['kmed'],
            scaler=data['scaler'],
            cache_path=data.get('cache_path')
        )

    @staticmethod
    def read(path: str) -> 'BreathingDataset':
        """
        Reads the BreathingDataset from a file.

        Args:
            path (str): Path to the file.

        Returns:
            BreathingDataset: The loaded dataset object.
        """
        try:
            with open(path, "rb") as file:
                data = pickle.load(file)
                return BreathingDataset.from_dict(data)
        except Exception as e:
            print(f"Error reading the file at {path}: {e}")
            raise

    def save(self, path: str):
        """
        Saves the BreathingDataset to a file.

        Args:
            path (str): Path to the file.
        """
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            with open(path, "wb") as file:
                pickle.dump(self.to_dict(), file)
            self.cache_path = path
        except Exception as e:
            print(f"Error saving the file at {path}: {e}")
            raise

    def set_cache_path(self, path: str):
        """
        Sets the cache path for the dataset.

        Args:
            path (str): Path to set as the cache path.
        """
        self.cache_path = path

    def load_from_cache(self):
        """
        Loads the dataset from the cache path if available.
        """
        if self.cache_path and os.path.exists(self.cache_path):
            self.read(self.cache_path)
