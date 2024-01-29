from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from src.config.dataclassess import TrainParams, MainParams
from src.models.utils import dfs2tensors
from src.models.window.forward import forward_sequences, pad_sequences
from src.nn.archs.window_lstm import WindowedConvLSTM
from src.nn.callbacks.early_stopping import EarlyStopping

from torch.functional import F

from src.nn.callbacks.schedules import LrSchedule


class WindowModel:
    def __init__(self,
                 main_params: MainParams,
                 n_attr_in: int,
                 window_size: int = 9,
                 ):
        self.n_attr_in = n_attr_in
        self.n_attr_out = len(main_params.cols_predict) if main_params.cols_predict is not None else n_attr_in
        self.main_params = main_params

        """
            net_params:
            device: 'cuda:0'
            save_path: 'models'
        
            conv_layer_data:
              - channels: 32
                kernel_size: 3
                activation: relu
        
              - channels: 32
                kernel_size: 3
                activation: relu
        
              - channels: 32
                kernel_size: 3
                activation: relu
        
            lstm_hidden_size: 128,
            #Add learning rate and other session_payload for network
        """
        self.model = WindowedConvLSTM(
            output_size=self.n_attr_out * main_params.n_steps_predict,
            device=main_params.device,
            n_attr=self.n_attr_in if main_params.cols_predict_training else self.n_attr_in - len(
                main_params.cols_predict),

            # Conv
            # conv_layers_data=params.conv_layer_data,

            # LSTM
            # lstm_hidden_size=params.lstm_hidden_size,
            lstm_layers=2,
            lstm_dropout=0.5,

            # MLP
            mlp_arch=[128, 128, 64, 16],
            mlp_dropout=0.5,
            mlp_activation=F.selu
        )

        self.model = self.model.to(self.main_params.device)

        self.criterion = None
        self.optimizer = None

        self.window_size = window_size

        self.scaler: Optional[MinMaxScaler] = None
        self.target_col_indexes = None

    def train(self, params: TrainParams, sequences: list[pd.DataFrame],
              val_perc: float = 0.2) -> Tuple[List[float], List[float]]:
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=0.001, lr=0.001)

        early_stopping = EarlyStopping(self.model, patience=params.es_patience)

        lr_schedule = LrSchedule(optimizer=self.optimizer, early_stopping=early_stopping, verbose=2)

        train_mae_losses = []
        test_mae_losses = []

        self.target_col_indexes = [sequences[0].columns.values.tolist().index(col) for col in
                                   self.main_params.cols_predict] \
            if self.main_params.cols_predict is not None else None

        train_sequences, val_sequences, (self.scaler, split) = dfs2tensors(sequences, val_perc=val_perc,
                                                                           device=self.main_params.device)

        # split.plot_split(title="Train and validation data plots", axe_titles=['a', 'b', 'std'])

        for epoch in range(params.epochs):
            print(f"Epoch {epoch + 1}/{params.epochs}\n")

            train_loss, mae_train_loss = forward_sequences(train_sequences,
                                                           is_eval=False,
                                                           model=self.model,
                                                           main_params=self.main_params,
                                                           optimizer=self.optimizer,
                                                           criterion=self.criterion,
                                                           target_indexes=self.target_col_indexes,
                                                           window_size=self.window_size
                                                           )

            val_loss, mae_val_loss = forward_sequences(val_sequences, is_eval=True,
                                                       model=self.model,
                                                       main_params=self.main_params,
                                                       optimizer=self.optimizer,
                                                       criterion=self.criterion,
                                                       target_indexes=self.target_col_indexes,
                                                       window_size=self.window_size)

            train_mae_losses.append(mae_train_loss)
            test_mae_losses.append(mae_val_loss)

            print(f"Train loss: {train_loss}, Val loss: {val_loss}")
            print(f"Train MAE: {mae_train_loss}, Val MAE: {mae_val_loss}")

            if early_stopping(val_loss):
                print("Early stopping")
                break

            lr_schedule.step()

        early_stopping.retrieve()

        return train_mae_losses, test_mae_losses

    def transform_y(self, data: np.ndarray):
        min_, max_ = self.scaler.feature_range
        data_range_ = self.scaler.data_range_[self.target_col_indexes]
        data_min_ = self.scaler.data_min_[self.target_col_indexes]

        data = (data - data_min_) / data_range_
        transformed = data * (max_ - min_) + min_

        return transformed

    def inverse_transform_y(self, data: np.ndarray):
        min_, max_ = self.scaler.feature_range
        inverse_data = (data - min_) / (max_ - min_)

        data_range_ = self.scaler.data_range_[self.target_col_indexes]
        data_min_ = self.scaler.data_min_[self.target_col_indexes]
        inverse_data = inverse_data * data_range_ + data_min_

        return inverse_data

    def predict(self, sequence_df: pd.DataFrame, return_inv_transformed: bool = True) -> torch.Tensor:
        sequence_array = sequence_df.values

        sequence_array = sequence_array[-self.window_size:]

        if self.main_params.cols_predict_training:
            sequence_array = self.scaler.transform(sequence_array)
        else:
            # NOTE: It doesn't work if we predict other column than respiration
            new_column = np.zeros((sequence_array.shape[0], 1))
            sequence_array_with_zeros = np.hstack((sequence_array, new_column))
            sequence_array_transformed = self.scaler.transform(sequence_array_with_zeros)
            sequence_array = np.delete(sequence_array_transformed, -1, axis=1)

        self.model.eval()

        with torch.no_grad():
            t = torch.Tensor(sequence_array).to(self.main_params.device)
            t = pad_sequences([t], window_size=self.window_size)[0]
            t = t.expand(1, *t.shape)

            out = self.model.forward(t, None)
            out = out.to('cpu')

        return self.inverse_transform_y(out) if return_inv_transformed else out
