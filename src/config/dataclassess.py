from dataclasses import dataclass
from typing import Optional, Callable, List, Literal

import torch
import torch.nn.functional as F

from src.nn.archs.lazy_mlc import MLConv, ConvLayerData as CLD

SUPPORTED_MODEL_TYPES = ['step', 'window']


@dataclass
class MainParams:
    model_type: str
    n_steps_predict: int
    cols_predict: List[str]

    device: torch.device
    cols_predict_training: bool = True

    def __repr__(self):
        return f"{self.model_type=}\n" \
               f"{self.n_steps_predict=}\n" \
               f"{self.cols_predict=}\n" \
               f"{self.cols_predict_training=}\n" \
               f"{self.device=}\n"


@dataclass
class TrainParams:
    """
    Parameters used for training
    :var es_patience: Early stopping patience
    :var epochs: Max number of epochs
    :var sequence_limit: Max number of sequences, used only for development purposes
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


SUPPORTED_TEST_MODES = ["full", "pessimistic", "both"]


@dataclass
class TestParams:
    """
    Parameters used for test
    """
    mode: str

    def __init__(self, mode: str):
        if mode not in SUPPORTED_TEST_MODES:
            raise ValueError(
                f"Mode \"{mode}\" is not a supported testing mode. Choose one of the following: {SUPPORTED_TEST_MODES}")

        self.mode = mode

    def __repr__(self):
        return f"{self.mode=}"
