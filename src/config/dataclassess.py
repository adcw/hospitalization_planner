from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn.functional as F

activation_dict = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "selu": F.selu
}


@dataclass
class ModelParams:
    """
    Class that contains all the data for neural network architecture
    """
    cols_predict: Optional[list[str]] = None
    n_steps_predict: int = 1

    lstm_hidden_size: int = 64
    n_lstm_layers: int = 2
    lstm_dropout: int = 0.2

    fccn_arch: Optional[list[int]] = None
    fccn_dropout_p: float = 0.15
    fccn_activation: Callable[[torch.Tensor, bool], torch.Tensor] = torch.nn.functional.relu

    device: torch.device = 'cpu'

    def __repr__(self):
        return f"{self.device=}\n" \
               f"{self.lstm_hidden_size=}\n" \
               f"{self.n_lstm_layers=}\n" \
               f"{self.n_steps_predict=}\n" \
               f"{self.lstm_dropout=}\n" \
               f"{self.fccn_arch=}\n" \
               f"{self.fccn_dropout_p=}\n" \
               f"{self.fccn_activation=}\n" \
               f"{self.cols_predict=}\n"


@dataclass
class TrainParams:
    """
    Parameters used for training
    :var es_patience: Early stopping patience
    :var epochs: Max number of epochs
    """
    es_patience: int = 2
    epochs: int = 30
    sequence_limit: Optional[int] = None

    def __repr__(self):
        return f"{self.es_patience=}\n" \
               f"{self.epochs=}\n" \
               f"{self.sequence_limit=}\n"


@dataclass
class EvalParams:
    """
    Parameters used for training
    :var es_patience: Early stopping patience
    :var epochs: Max number of epochs
    """
    es_patience: int = 2
    epochs: int = 30
    n_splits: int = 5
    sequence_limit: Optional[int] = None

    def __repr__(self):
        return f"{self.es_patience=}\n" \
               f"{self.epochs=}\n" \
               f"{self.n_splits=}\n" \
               f"{self.sequence_limit=}\n"
