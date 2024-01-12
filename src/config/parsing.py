from typing import Tuple

import torch
import torch.nn.functional as F
import yaml
from src.nn.archs.lazy_mlc import MLConv, ConvLayerData as CLD

from src.config.dataclassess import StepModelParams, TrainParams, EvalParams, activation_dict, WindowModelParams, \
    TestParams


def _parse_step_params(data):
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

    return StepModelParams(device=device,
                           lstm_hidden_size=lstm_hidden_size,
                           n_lstm_layers=n_lstm_layers,
                           n_steps_predict=n_steps_predict,
                           cols_predict=cols_predict,

                           fccn_arch=fccn_arch,
                           fccn_dropout_p=fccn_dropout_p,
                           fccn_activation=fccn_activation,

                           lstm_dropout=lstm_dropout,
                           save_path=save_path)


def _parse_window_params(data) -> WindowModelParams:
    # Extract model parameters
    model_params_data = data['model_params']['net_params']

    n_steps_predict = data['model_params']['n_steps_predict']
    cols_predict = data['model_params']['cols_predict']
    save_path = model_params_data['save_path']

    conv_layer_data_raw = model_params_data['conv_layer_data']
    # lstm_hidden_size = model_params_data['lstm_hidden_size']

    # conv_layer_data = [
    #     CLD(channels=entry['channels'], kernel_size=entry['kernel_size'],
    #         activation=activation_dict.get(entry['activation'], F.relu))
    #     for entry in conv_layer_data_raw
    # ]

    return WindowModelParams(
        n_steps_predict=n_steps_predict,
        cols_predict=cols_predict,
        save_path=save_path,

        # conv_layer_data=conv_layer_data,
        # lstm_hidden_size=lstm_hidden_size
    )


def parse_config(yaml_path: str) -> Tuple[StepModelParams | WindowModelParams, TrainParams, EvalParams, TestParams]:
    with open(yaml_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)

    model_type = data['model_type']

    if model_type == 'step':
        model_params = _parse_step_params(data)
    elif model_type == 'window':
        model_params = _parse_window_params(data)
        pass
    else:
        raise ValueError(f"Model type \"{model_type}\" is not supported.")

    # Extract training parameters
    train_params_data = data['train']
    es_patience = train_params_data['es_patience']
    epochs = train_params_data['epochs']
    sequence_limit = train_params_data['sequence_limit']

    train_params = TrainParams(es_patience=es_patience,
                               epochs=epochs,
                               sequence_limit=sequence_limit)

    # Extract eval parameters
    eval_params_data = data['eval']
    es_patience = eval_params_data['es_patience']
    epochs = eval_params_data['epochs']
    sequence_limit = eval_params_data['sequence_limit']
    n_splits = eval_params_data['n_splits']

    eval_params = EvalParams(es_patience=es_patience,
                             epochs=epochs,
                             sequence_limit=sequence_limit,
                             n_splits=n_splits)

    # Extract test parameters
    test_params_data = data['test']
    test_mode = test_params_data['mode']

    test_params = TestParams(mode=test_mode)

    return model_params, train_params, eval_params, test_params
