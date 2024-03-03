from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from src.config.dataclassess import MainParams
from src.config.parsing import TrainParams
from src.models.step.forward import forward_sequences
from src.models.utils import dfs2tensors
from src.nn.archs.step_time_lstm import StepTimeLSTM
from src.nn.callbacks.early_stopping import EarlyStopping
from src.nn.callbacks.schedules import LrSchedule


class StepModel:
    def __init__(self,
                 main_params: MainParams,
                 n_attr_in: int,
                 ):
        """

        :param main_params: Params read from config file
        """
        self.n_attr_out = len(main_params.cols_predict) if main_params.cols_predict is not None else n_attr_in
        self.main_params = main_params

        # n_attr_in = n_attr_in if main_params.cols_predict_training else n_attr_in - len(main_params.cols_predict)

        self.model = StepTimeLSTM(input_size=n_attr_in,
                                  output_size=self.n_attr_out * self.main_params.n_steps_predict,
                                  device=self.main_params.device, )

        self.model = self.model.to(self.main_params.device)

        self.criterion = None
        self.optimizer = None

        self.scaler: Optional[MinMaxScaler] = None
        self.target_col_indexes = None

    def train(self, params: TrainParams, sequences: list[pd.DataFrame],
              val_perc: float = 0.2) -> Tuple[List[float], List[float]]:
        """
        Train the model

        :param plot:
        :param params: Training parameters
        :param sequences: A list of sequences to be learned
        :return: Final training loss
        """
        self.criterion = nn.MSELoss()
        # self.criterion = nn.HuberLoss(reduction='mean', delta=0.125)
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=0.001, lr=0.001)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)

        early_stopping = EarlyStopping(self.model, patience=params.es_patience)
        lr_schedule = LrSchedule(optimizer=self.optimizer, early_stopping=early_stopping, verbose=2)

        train_abs_losses = []
        val_abs_losses = []

        self.target_col_indexes = [sequences[0].columns.values.tolist().index(col) for col in
                                   self.main_params.cols_predict] \
            if self.main_params.cols_predict is not None else None

        train_sequences, val_sequences, (scaler, split) = dfs2tensors(sequences,
                                                                      val_perc=val_perc,
                                                                      device=self.main_params.device)

        # split.plot_split(title="Train and validation sequences")
        for epoch in range(params.epochs):
            print(f"Epoch {epoch + 1}/{params.epochs}\n")

            # Forward test data
            train_sqrt_loss, train_abs_loss, _ = forward_sequences(train_sequences, is_eval=False,
                                                                   model=self.model,
                                                                   main_params=self.main_params,
                                                                   optimizer=self.optimizer,
                                                                   criterion=self.criterion,
                                                                   target_indexes=self.target_col_indexes,
                                                                   y_cols_in_x=self.main_params.cols_predict_training)

            # Forward val data
            val_sqrt_loss, val_abs_loss, _ = forward_sequences(val_sequences, is_eval=True,
                                                               model=self.model,
                                                               main_params=self.main_params,
                                                               optimizer=self.optimizer,
                                                               criterion=self.criterion,
                                                               target_indexes=self.target_col_indexes,
                                                               y_cols_in_x=self.main_params.cols_predict_training)

            train_abs_losses.append(train_abs_loss)
            val_abs_losses.append(val_abs_loss)

            print(f"Train loss: {train_sqrt_loss}, Val loss: {val_sqrt_loss}")
            print(f"Train MAE: {train_abs_loss}, Val MAE: {val_abs_loss}")

            if early_stopping(val_sqrt_loss):
                print("Early stopping")
                break

            lr_schedule.step()

        early_stopping.retrieve()
        self.scaler = scaler

        return train_abs_losses, val_abs_losses

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
        """

        :param sequence_df:
        :param return_inv_transformed: Work on already transformed data
        :return: Return inverse transformed and raw network output.
        """
        sequence_array = sequence_df.values

        if self.main_params.cols_predict_training:
            sequence_array = self.scaler.transform(sequence_array)
        else:
            # NOTE: It doesn't work if we predict other column than respiration
            new_column = np.zeros((sequence_array.shape[0], 1))
            sequence_array_with_zeros = np.hstack((sequence_array, new_column))
            sequence_array_transformed = self.scaler.transform(sequence_array_with_zeros)
            sequence_array = np.delete(sequence_array_transformed, -1, axis=1)

        self.model.eval()

        # Initialize hidden states
        h0 = torch.zeros((self.model.lstm_num_layers, self.model.lstm_hidden_size),
                         device=self.main_params.device)
        c0 = torch.zeros((self.model.lstm_num_layers, self.model.lstm_hidden_size),
                         device=self.main_params.device)

        out = None

        with torch.no_grad():
            for step in sequence_array:
                step_tensor = torch.Tensor(step).expand((1, -1)).to(self.main_params.device)

                out, (hn, cn) = self.model.forward(step_tensor, h0, c0)

                h0 = hn.detach()
                c0 = cn.detach()

        out = out.to('cpu')

        return self.inverse_transform_y(out) if return_inv_transformed else out
