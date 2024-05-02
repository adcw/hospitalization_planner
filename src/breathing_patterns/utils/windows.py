from typing import List, Optional

import pandas as pd

from src.tools.iterators import windowed


def make_windows(seqs: List[pd.DataFrame], window_size: int, stride: int):
    windows = []

    for seq in seqs:
        windows.extend([w for w in windowed(seq, window_size=window_size, stride=stride)])

    return windows


def xy_windows_split(windows: List, target_len: int, min_x_len: Optional[int] = None):
    if min_x_len is None:
        min_x_len = target_len

    x_windows = []
    y_windows = []
    for w in windows:
        w_len = len(w)
        if w_len < min_x_len + target_len:
            continue

        x_windows.append(w[:-target_len])
        y_windows.append(w[-target_len:])

    return x_windows, y_windows
