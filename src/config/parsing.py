from dataclasses import dataclass
from typing import Optional, Callable, Tuple

import torch
import torch.nn.functional as F
import yaml

...

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


def parse_config(yaml_path: str) -> Tuple[ModelParams, TrainParams, EvalParams]:
    with open(yaml_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)

    # Extract model parameters
    model_params_data = data['model_params']['net_params']
    device = torch.device(model_params_data['device'])

    lstm_hidden_size = model_params_data['lstm_hidden_size']
    n_lstm_layers = model_params_data['n_lstm_layers']
    lstm_dropout = model_params_data['lstm_dropout']

    n_steps_predict = data['model_params']['n_steps_predict']
    cols_predict = data['model_params']['cols_predict']

    fccn_arch = model_params_data['fccn_arch']
    fccn_dropout_p = model_params_data['fccn_dropout_p']
    fccn_activation_name = model_params_data['fccn_activation']
    fccn_activation = activation_dict.get(fccn_activation_name, F.relu)

    model_params = ModelParams(device=device, lstm_hidden_size=lstm_hidden_size, n_lstm_layers=n_lstm_layers,
                               n_steps_predict=n_steps_predict, cols_predict=cols_predict, fccn_arch=fccn_arch,
                               fccn_dropout_p=fccn_dropout_p, fccn_activation=fccn_activation,
                               lstm_dropout=lstm_dropout)

    # Extract training parameters
    train_params_data = data['train']
    es_patience = train_params_data['es_patience']
    epochs = train_params_data['epochs']
    sequence_limit = train_params_data['sequence_limit']

    train_params = TrainParams(es_patience=es_patience, epochs=epochs,
                               sequence_limit=sequence_limit)

    # Extract eval parameters
    eval_params_data = data['eval']
    es_patience = eval_params_data['es_patience']
    epochs = eval_params_data['epochs']
    sequence_limit = eval_params_data['sequence_limit']
    n_splits = eval_params_data['n_splits']

    eval_params = EvalParams(es_patience=es_patience, epochs=epochs, sequence_limit=sequence_limit, n_splits=n_splits)

    return model_params, train_params, eval_params
