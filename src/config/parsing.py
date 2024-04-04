from typing import Tuple

import yaml

from src.config.dataclassess import MainParams, TrainParams, EvalParams, TestParams, SUPPORTED_MODEL_TYPES


def parse_config(yaml_path: str) -> Tuple[MainParams, TrainParams, EvalParams, TestParams]:
    with open(yaml_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)

    # model_type = data['model_type']
    #
    # if model_type == 'step':
    #     model_params = _parse_step_params(data)
    # elif model_type == 'window':
    #     model_params = _parse_window_params(data)
    #     pass
    # else:
    #     raise ValueError(f"Model type \"{model_type}\" is not supported.")

    # Extract main params
    main_params_data = data['main']
    model_type = main_params_data['model_type']

    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"{model_type} is not supported model type. Chose one of: {SUPPORTED_MODEL_TYPES}")

    n_steps_predict = main_params_data['n_steps_predict']
    cols_predict = main_params_data['cols_predict']
    device = main_params_data['device']
    cols_predict_training = main_params_data['cols_predict_training']

    main_params = MainParams(model_type=model_type,
                             n_steps_predict=n_steps_predict,
                             cols_predict=cols_predict,
                             cols_predict_training=cols_predict_training,
                             device=device,

                             test_size=main_params_data['test_size'],
                             val_size=main_params_data['val_size'])

    # Extract training parameters
    train_params_data = data['train']
    es_patience = train_params_data['es_patience']
    epochs = train_params_data['epochs']
    sequence_limit = train_params_data['sequence_limit']

    train_params = TrainParams(es_patience=es_patience,
                               epochs=epochs,
                               sequence_limit=sequence_limit,
                               batch_size=train_params_data['batch_size'])

    # Extract eval parameters
    eval_params_data = data['eval']
    es_patience = eval_params_data['es_patience']
    epochs = eval_params_data['epochs']
    sequence_limit = eval_params_data['sequence_limit']
    n_splits = eval_params_data['n_splits']

    eval_params = EvalParams(es_patience=es_patience,
                             epochs=epochs,
                             sequence_limit=sequence_limit,
                             n_splits=n_splits,
                             batch_size=eval_params_data['batch_size'])

    # Extract test parameters
    test_params_data = data['test']
    test_mode = test_params_data['mode']

    test_params = TestParams(mode=test_mode)

    return main_params, train_params, eval_params, test_params
