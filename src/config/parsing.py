from typing import Tuple

import torch
import torch.nn.functional as F
import yaml

from src.config.dataclassess import ModelParams, TrainParams, EvalParams, activation_dict


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

    save_path = model_params_data['save_path']

    model_params = ModelParams(device=device, lstm_hidden_size=lstm_hidden_size, n_lstm_layers=n_lstm_layers,
                               n_steps_predict=n_steps_predict, cols_predict=cols_predict, fccn_arch=fccn_arch,
                               fccn_dropout_p=fccn_dropout_p, fccn_activation=fccn_activation,
                               lstm_dropout=lstm_dropout, save_path=save_path)

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
