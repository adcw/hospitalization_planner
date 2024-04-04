from dataclasses import dataclass
from typing import Optional, List

import torch

SUPPORTED_MODEL_TYPES = ['step', 'window']
SUPPORTED_TEST_MODES = ['full', 'pessimistic', 'both']


@dataclass
class MainParams:
    model_type: str
    n_steps_predict: int
    cols_predict: List[str]

    device: torch.device
    cols_predict_training: bool = True

    test_size: float = 0.1
    val_size: float = 0.1

    def __repr__(self):
        return f"{self.model_type=}\n" \
               f"{self.n_steps_predict=}\n" \
               f"{self.cols_predict=}\n" \
               f"{self.device=}\n" \
               f"{self.cols_predict_training=}\n" \
               f"{self.test_size=}\n" \
               f"{self.val_size=}\n"


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
    batch_size: int = 16
    sequence_limit: Optional[int] = None

    def __repr__(self):
        return f"{self.es_patience=}\n" \
               f"{self.epochs=}\n" \
               f"{self.batch_size=}\n" \
               f"{self.sequence_limit=}\n"


@dataclass
class EvalParams:
    """
    Parameters used for evaluation
    :var es_patience: Early stopping patience
    :var epochs: Max number of epochs
    """
    es_patience: int = 2
    epochs: int = 30
    n_splits: int = 5
    sequence_limit: Optional[int] = None
    batch_size: int = 16

    def __repr__(self):
        return f"{self.es_patience=}\n" \
               f"{self.epochs=}\n" \
               f"{self.n_splits=}\n" \
               f"{self.sequence_limit=}\n" \
               f"{self.batch_size=}\n"


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
