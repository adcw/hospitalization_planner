from dataclasses import dataclass
from typing import Optional, Callable, Tuple

import torch
import torch.nn.functional as F
import yaml

...

activation_dict = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
}


@dataclass
class ModelParams:
    """
    Class that contains all the data for neural network architecture
    """
    device: torch.device
    hidden_size: int = 64
    n_lstm_layers: int = 2
    n_steps_predict: int = 1
    cols_predict: Optional[list[str]] = None,
    fccn_arch: list[int] = [32] * 5,
    fccn_dropout_p: float = 0.15
    fccn_activation: Callable[[torch.Tensor, bool], torch.Tensor] = torch.nn.functional.relu


@dataclass
class TrainParams:
    """
    Parameters used for training
    :var es_patience: Early stopping patience
    :var epochs: Max number of epochs
    :var eval_n_splits: Number of kfold splits used in validation
    """
    es_patience: int = 2
    epochs: int = 30
    eval_n_splits: int = 5


def parse_config(yaml_path: str) -> Tuple[ModelParams, TrainParams]:
    with open(yaml_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)

    # Extract model parameters
    model_params_data = data['model_params']['net_params']
    device = torch.device(model_params_data['device'])
    hidden_size = model_params_data['hidden_size']
    n_lstm_layers = model_params_data['n_lstm_layers']
    n_steps_predict = data['model_params']['n_steps_predict']
    cols_predict = data['model_params']['cols_predict']
    fccn_arch = model_params_data['fccn_arch']
    fccn_dropout_p = model_params_data['fccn_dropout_p']
    fccn_activation_name = model_params_data['fccn_activation']
    fccn_activation = activation_dict.get(fccn_activation_name, F.relu)

    model_params = ModelParams(device=device, hidden_size=hidden_size, n_lstm_layers=n_lstm_layers,
                               n_steps_predict=n_steps_predict, cols_predict=cols_predict, fccn_arch=fccn_arch,
                               fccn_dropout_p=fccn_dropout_p, fccn_activation=fccn_activation)

    # Extract training parameters
    train_params_data = data['train']
    es_patience = train_params_data['es_patience']
    epochs = train_params_data['epochs']
    eval_n_splits = train_params_data['eval_n_splits']

    train_params = TrainParams(es_patience=es_patience, epochs=epochs, eval_n_splits=eval_n_splits)

    return model_params, train_params
