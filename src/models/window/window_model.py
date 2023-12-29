from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from src.config.dataclassess import ModelParams, TrainParams
from src.models.utils import dfs2tensors
from src.models.window.forward import forward_sequences, pad_sequences
from src.nn.archs.window_lstm import WindowedConvLSTM
from src.nn.callbacks.early_stopping import EarlyStopping


class WindowModel:
    def __init__(self,
                 params: ModelParams,
                 n_attr_in: int,
                 window_size: int = 10,
                 ):
        self.n_attr_in = n_attr_in
        self.n_attr_out = len(params.cols_predict) if params.cols_predict is not None else n_attr_in
        self.model_params = params

        # self.model = WindowedLSTM(input_size=self.n_attr_in,
        #                           output_size=self.n_attr_out * self.model_params.n_steps_predict,
        #                           device=self.model_params.device)

        self.model = WindowedConvLSTM(
            output_size=self.n_attr_out * self.model_params.n_steps_predict,
            device=self.model_params.device,
            n_attr=self.n_attr_in,

            lstm_hidden_size=32,
            lstm_layers=2,
            lstm_dropout=0.2,

            conv_kernel_size=5,
            conv_stride=1,
            conv_channels=32,

            mlp_arch=[128, 64, 16],
            mlp_dropout=0.2
        )

        self.model = self.model.to(self.model_params.device)

        self.criterion = None
        self.optimizer = None

        self.window_size = window_size

        self.scaler: Optional[MinMaxScaler] = None
        self.target_col_indexes = None

    def train(self, params: TrainParams, sequences: list[pd.DataFrame], plot: bool = True,
              val_perc: float = 0.2) -> Tuple[float, float]:
        self.criterion = nn.MSELoss()
        # self.criterion = nn.HuberLoss(reduction='mean', delta=0.125)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=3e-4)

        early_stopping = EarlyStopping(self.model, patience=params.es_patience)

        train_losses = []
        val_losses = []

        self.target_col_indexes = [sequences[0].columns.values.tolist().index(col) for col in
                                   self.model_params.cols_predict] \
            if self.model_params.cols_predict is not None else None

        train_sequences, val_sequences, (self.scaler, split) = dfs2tensors(sequences, val_perc=val_perc,
                                                                           device=self.model_params.device)

        split.plot_split(title="Train and validation data plots", axe_titles=['a', 'b', 'std'])

        for epoch in range(params.epochs):
            print(f"Epoch {epoch + 1}/{params.epochs}\n")

            train_loss, mae_train_loss = forward_sequences(train_sequences, is_eval=False,
                                                           model=self.model,
                                                           model_params=self.model_params,
                                                           optimizer=self.optimizer,
                                                           criterion=self.criterion,
                                                           target_indexes=self.target_col_indexes,
                                                           window_size=self.window_size
                                                           )

            val_loss, mae_val_loss = forward_sequences(val_sequences, is_eval=True,
                                                       model=self.model,
                                                       model_params=self.model_params,
                                                       optimizer=self.optimizer,
                                                       criterion=self.criterion,
                                                       target_indexes=self.target_col_indexes,
                                                       window_size=self.window_size)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Train loss: {train_loss}, Val loss: {val_loss}")
            print(f"Train MAE: {mae_train_loss}, Val MAE: {mae_val_loss}")

            if early_stopping(val_loss):
                print("Early stopping")
                break

        early_stopping.retrieve()

        # TODO: Save plots to directory
        if plot:
            plt.plot(train_losses, label="Train losses")
            plt.plot(val_losses, label="Val losses")
            plt.legend()
            plt.show()

        return train_losses[-1], val_losses[-1]

    def data_transform(self, data: np.ndarray):
        min_, max_ = self.scaler.feature_range
        data_range_ = self.scaler.data_range_[self.target_col_indexes]
        data_min_ = self.scaler.data_min_[self.target_col_indexes]

        data = (data - data_min_) / data_range_
        transformed = data * (max_ - min_) + min_

        return transformed

    def data_inverse_transform(self, data: np.ndarray):
        min_, max_ = self.scaler.feature_range
        inverse_data = (data - min_) / (max_ - min_)

        data_range_ = self.scaler.data_range_[self.target_col_indexes]
        data_min_ = self.scaler.data_min_[self.target_col_indexes]
        inverse_data = inverse_data * data_range_ + data_min_

        return inverse_data

    def predict(self, sequence_df: pd.DataFrame, return_inv_transformed: bool = True) -> torch.Tensor:
        sequence_array = sequence_df.values

        sequence_array = sequence_array[-self.window_size:]
        sequence_array = self.scaler.transform(sequence_array)

        self.model.eval()

        with torch.no_grad():
            # TODO: Calculate the maks
            t = torch.Tensor(sequence_array).to(self.model_params.device)
            t = pad_sequences([t], window_size=self.window_size)[0]
            t = t.expand(1, *t.shape)

            out = self.model.forward(t, None)
            out = out.to('cpu')

        return self.data_inverse_transform(out) if return_inv_transformed else out
