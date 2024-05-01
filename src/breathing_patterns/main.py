from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.models.utils import dfs2tensors
from src.session.model_manager import _get_sequences
from src.tools.dataframe_scale import scale
from src.tools.iterators import windowed

# import data.colnames_original as c

CSV_PATH = './../../data/input.csv'
WINDOW_SIZE = 15
STRIDE_RATE = 0.2

if __name__ == '__main__':
    sequences, preprocessor = _get_sequences(path=CSV_PATH, limit=100)

    sequences_scaled, scaler = scale(sequences)

    stride = max(1, round(WINDOW_SIZE * STRIDE_RATE))

    windows = []

    for seq in sequences_scaled:
        windows.extend([w for w in windowed(seq, window_size=WINDOW_SIZE, stride=stride)])

    pass
