from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.model_selection.regression_strat_kfold import RegressionStratKFold
from src.models.step.step_model import StepModel
from src.models.window.window_model import WindowModel
from src.session.helpers.session_payload import SessionPayload
from src.session.helpers.test import test_model
from src.session.helpers.train import train_model


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

    train_losses = []
    val_losses = []
    test_losses = []

    for split_i, (train_index, val_index) in enumerate(kf.split(sequences)):
        print(f"Training on split number {split_i + 1}")

        model = WindowModel(payload.model_params, n_attr_in=sequences[0].shape[1])

        # Get train and validation tensors
        train_sequences = [sequences[i] for i in train_index]
        val_sequences = [sequences[i] for i in val_index]

        # Train on sequences
        train_loss, val_loss = model.train(train_params, train_sequences, plot=False)
        model_payload = deepcopy(payload)
        model_payload.model = model

        # Perform test
        test_loss = test_model(model_payload, val_sequences, limit=None, plot=False, max_per_sequence=None)

        test_model(model_payload, sequences=val_sequences, limit=30)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        print(f"Mean test loss: {test_loss}")

    plt.plot(train_losses, 'o', label="train_loss")
    plt.plot(val_losses, 'o', label="val_loss")
    plt.plot(test_losses, 'o', label="test_loss")

    # Adding text labels near data markers

    for i, value in enumerate(train_losses):
        plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')

    for i, value in enumerate(val_losses):
        plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')

    for i, value in enumerate(test_losses):
        s = f'{value:.4f}' if value is not None else ""
        plt.text(i, value, s, ha='center', va='bottom')

    plt.title(f"Losses on each fold. Avg = {np.average(test_losses)}")
    plt.legend()
    plt.grid(True)
    plt.show()
