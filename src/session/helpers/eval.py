from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config.dataclassess import WindowModelParams, StepModelParams
from src.model_selection.regression_strat_kfold import RegressionStratKFold
from src.models.step.step_model import StepModel
from src.models.window.window_model import WindowModel
from src.session.helpers.session_payload import SessionPayload
from src.session.helpers.test import test_model
from src.session.helpers.train import train_model
from src.session.utils.save_plots import save_plot


def eval_model(
        payload: SessionPayload,
        sequences: list[pd.DataFrame],
):
    # Retrieve session_payload, prepare training session_payload to perform CV
    eval_params = payload.eval_params
    train_params = deepcopy(payload.train_params)
    train_params.epochs = eval_params.epochs
    train_params.es_patience = eval_params.es_patience

    sequences = sequences[:payload.eval_params.sequence_limit]

    kf = RegressionStratKFold()

    split_train_losses = []
    split_val_losses = []
    split_test_losses = []

    for split_i, (train_index, val_index) in enumerate(kf.split(sequences)):
        print(f"Training on split number {split_i + 1}")

        if type(payload.model_params) == WindowModelParams:
            model = WindowModel(payload.model_params, n_attr_in=sequences[0].shape[1])
        elif type(payload.model_params) == StepModelParams:
            model = StepModel(payload.model_params, n_attr_in=sequences[0].shape[1])
        else:
            raise TypeError(f"Unknown param type: {type(payload.model_params)}")

        # Get train and validation tensors
        train_sequences = [sequences[i] for i in train_index]
        val_sequences = [sequences[i] for i in val_index]

        # Train on sequences
        train_losses, val_losses = model.train(train_params, train_sequences)

        plt.plot(train_losses, label="Train losses")
        plt.plot(val_losses, label="Val losses")
        plt.legend()
        save_plot(f"split_{split_i + 1}/loss.png")

        model_payload = deepcopy(payload)
        model_payload.model = model

        # Perform test
        test_loss = test_model(model_payload, val_sequences, plot=True)
        plt.subplots_adjust(top=0.95)
        plt.suptitle(f"MAE Test loss: {test_loss}", fontsize=20)
        save_plot(f"split_{split_i + 1}/preds.png")

        train_loss = train_losses[-1]
        val_loss = val_losses[-1]

        split_train_losses.append(train_loss)
        split_val_losses.append(val_loss)
        split_test_losses.append(test_loss)

        print(f"Mean test loss: {test_loss}")

    plt.plot(split_train_losses, 'o', label="train_loss")
    plt.plot(split_val_losses, 'o', label="val_loss")
    plt.plot(split_test_losses, 'o', label="test_loss")

    # Adding text labels near data markers
    for i, value in enumerate(split_train_losses):
        plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')

    for i, value in enumerate(split_val_losses):
        plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')

    for i, value in enumerate(split_test_losses):
        s = f'{value:.4f}' if value is not None else ""
        plt.text(i, value, s, ha='center', va='bottom')

    plt.title(f"Losses on each fold. Avg test loss = {np.average(split_test_losses)}")
    plt.legend()
    plt.grid(True)

    save_plot(f"fold_losses.png")
