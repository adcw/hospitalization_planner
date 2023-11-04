from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn.functional as F
import yaml


@dataclass
class Params(ABC):
    @classmethod
    @abstractmethod
    def from_yaml(cls, yaml_path):
        ...


activation_dict = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
}


@dataclass
class ModelParams(Params):
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

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)

        device = torch.device(data['model_params']['net_params']['device'])
        hidden_size = data['model_params']['net_params']['hidden_size']
        n_lstm_layers = data['model_params']['net_params']['n_lstm_layers']
        n_steps_predict = data['model_params']['n_steps_predict']
        cols_predict = data['model_params']['cols_predict']
        fccn_arch = data['model_params']['net_params']['fccn_arch']
        fccn_dropout_p = data['model_params']['net_params']['fccn_dropout_p']

        fccn_activation_name = data['model_params']['net_params']['fccn_activation']

        fccn_activation = activation_dict.get(fccn_activation_name, F.relu)

        return cls(device=device, hidden_size=hidden_size, n_lstm_layers=n_lstm_layers, n_steps_predict=n_steps_predict,
                   cols_predict=cols_predict, fccn_arch=fccn_arch, fccn_dropout_p=fccn_dropout_p,
                   fccn_activation=fccn_activation)


@dataclass
class TrainParams(Params):
    """
    Parameters used for training
    :var es_patience: Early stopping patience
    :var epochs: Max number of epochs
    :var eval_n_splits: Number of kfold splits used in validation
    """
    es_patience: int = 2
    epochs: int = 30
    eval_n_splits: int = 5

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)

        es_patience = data['train']['es_patience']
        epochs = data['train']['epochs']
        eval_n_splits = data['train']['eval_n_splits']

        return cls(es_patience, epochs, eval_n_splits)
