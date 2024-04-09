from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model_selection.regression_strat_kfold import RegressionStratKFold
from src.model_selection.regression_train_test_split import RegressionTrainTestSplitter
from src.model_selection.stratified_sampling import stratified_sampling
from src.models.step.step_model import StepModel
from src.models.window.window_model import WindowModel
from src.session.helpers.session_payload import SessionPayload
from src.session.helpers.test import test_model, test_model_state_optimal
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
    split_test_losses_full = []
    split_test_losses_pessimistic = []

    for split_i, (train_index, test_index) in enumerate(kf.split(sequences)):
        print(f"Training on split number {split_i + 1}")

        if payload.main_params.model_type == 'window':
            model = WindowModel(payload.main_params, n_attr_in=sequences[0].shape[1])

        elif payload.main_params.model_type == 'step':
            model = StepModel(payload.main_params, n_attr_in=sequences[0].shape[1])
        else:
            raise TypeError(f"Unknown model type: {type(payload.main_params.model_type)}")

        # Get train and validation tensors
        train_sequences = [sequences[i] for i in train_index]
        test_sequences = [sequences[i] for i in test_index]
        train_sequences, val_sequences = RegressionTrainTestSplitter().fit_split(train_sequences,
                                                                                 test_size=payload.main_params.val_size)

        # Train on sequences
        train_losses, val_losses, last_train_loss, last_val_loss = model.train(train_params, val_sequences)

        plt.plot(train_losses, label="Train losses")
        plt.plot(val_losses, label="Val losses")
        plt.legend()
        save_plot(f"split_{split_i + 1}/loss.png")

        model_payload = deepcopy(payload)
        model_payload.model = model

        # Perform test
        plot_indexes = stratified_sampling(kf.clusters[test_index], 12)

        if payload.main_params.model_type == "step":
            full_loss = test_model_state_optimal(model_payload, test_sequences, plot=True, plot_indexes=plot_indexes)
        elif payload.main_params.model_type == "window":
            full_loss, _ = test_model(model_payload, test_sequences, plot=True, plot_indexes=plot_indexes)
        else:
            raise ValueError(f"Unsupported model type: {payload.main_params.model_type}")

        plt.subplots_adjust(top=0.95)
        plt.suptitle(f"MAE Test loss: {full_loss}", fontsize=20)
        save_plot(f"split_{split_i + 1}/test_full.png")

        pessimistic_loss, _ = test_model(model_payload, test_sequences, plot=True, plot_indexes=plot_indexes,
                                         mode='pessimistic')

        plt.subplots_adjust(top=0.95)
        plt.suptitle(f"MAE Test loss: {pessimistic_loss}", fontsize=20)
        save_plot(f"split_{split_i + 1}/test_pessimistic.png")

        split_train_losses.append(last_train_loss)
        split_val_losses.append(last_val_loss)
        split_test_losses_full.append(full_loss)
        split_test_losses_pessimistic.append(pessimistic_loss)

        print(f"Mean test loss: {full_loss}")

    plt.plot(split_train_losses, 'o', label=f"train_loss, mean={np.mean(split_train_losses):.4f}")
    plt.plot(split_val_losses, 'o', label=f"val_loss, mean={np.mean(split_val_losses):.4f}")
    plt.plot(split_test_losses_full, 'o', label=f"full_loss, mean={np.mean(split_test_losses_full):.4f}")
    plt.plot(split_test_losses_pessimistic, 'o',
             label=f"pessimistic loss, mean={np.mean(split_test_losses_pessimistic):.4f}")

    # Adding text labels near data markers
    for i, value in enumerate(split_train_losses):
        plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')

    for i, value in enumerate(split_val_losses):
        plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')

    for i, value in enumerate(split_test_losses_full):
        s = f'{value:.4f}' if value is not None else ""
        plt.text(i, value, s, ha='center', va='bottom')

    for i, value in enumerate(split_test_losses_pessimistic):
        s = f'{value:.4f}' if value is not None else ""
        plt.text(i, value, s, ha='center', va='bottom')

    plt.title(f"Losses on each fold")
    plt.legend()
    plt.grid(True)

    save_plot(f"fold_losses.png")
