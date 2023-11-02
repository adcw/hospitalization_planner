from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import yaml


@dataclass
class Params(ABC):
    @classmethod
    @abstractmethod
    def from_yaml(cls, yaml_path):
        ...


@dataclass
class NetParams(Params):
    """
    Class that contains all the data for neural network architecture
    """
    n_attr: int
    device: torch.device
    hidden_size: int = 64
    n_lstm_layers: int = 2

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)

        n_attr = data['net_params']['n_attr']
        device = torch.device(data['net_params']['device'])
        hidden_size = data['net_params']['hidden_size']
        n_lstm_layers = data['net_params']['n_lstm_layers']

        return cls(n_attr=n_attr, device=device, hidden_size=hidden_size, n_lstm_layers=n_lstm_layers)


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
