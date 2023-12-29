from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src.model_selection.regression_train_test_split import RegressionTrainTestSplitter
from src.preprocessing.normalization.utils import split_and_norm_sequences


def seq2tensors(sequences: list[np.ndarray], device: torch.device) -> List[torch.Tensor]:
    tensors = []
    for seq in sequences:
        tensor = torch.Tensor(seq)
        tensor = tensor.to(device)
        tensors.append(tensor)
    return tensors


def dfs2tensors(dfs: List[pd.DataFrame],
                val_perc: Optional[float] = 0.2,
                device: torch.device = "cuda"
                ) \
        -> Tuple[
            List[torch.Tensor],
            Optional[List[torch.Tensor]],
            Tuple[MinMaxScaler, RegressionTrainTestSplitter]
        ]:
    sequences = [s.values for s in dfs]

    train_sequences, val_sequences, scaler = split_and_norm_sequences(sequences, val_perc)
    train_sequences = seq2tensors(train_sequences, device)

    val_sequences = seq2tensors(val_sequences, device) if len(val_sequences) != 0 else []

    return train_sequences, val_sequences, scaler
