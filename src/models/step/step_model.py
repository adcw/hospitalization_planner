from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from src.config.parsing import ModelParams, TrainParams
from src.models.step.forward import forward_sequences
from src.models.utils import dfs2tensors
from src.nn.archs.step_time_lstm import StepTimeLSTM
from src.nn.callbacks.early_stopping import EarlyStopping


class StepModel:
    def __init__(self,
                 params: ModelParams,
                 n_attr_in: int,
                 ):
        """

        :param params: Params read from config file
        """
        self.n_attr_in = n_attr_in
        self.n_attr_out = len(params.cols_predict) if params.cols_predict is not None else n_attr_in
        self.model_params = params

        self.model = StepTimeLSTM(input_size=self.n_attr_in,
                                  lstm_hidden_size=self.model_params.lstm_hidden_size,
                                  n_lstm_layers=self.model_params.n_lstm_layers,
                                  output_size=self.n_attr_out * self.model_params.n_steps_predict,
                                  device=self.model_params.device,
                                  fccn_arch=self.model_params.fccn_arch,
                                  fccn_dropout_p=self.model_params.fccn_dropout_p)

        self.model = self.model.to(self.model_params.device)

        self.criterion = None
        self.optimizer = None

        self.scaler: Optional[MinMaxScaler] = None
        self.target_col_indexes = None

    def train(self, params: TrainParams, sequences: list[pd.DataFrame], plot: bool = True,
              val_perc: float = 0.2) -> Tuple[float, float]:
        """
        Train the model

        :param plot:
        :param params: Training parameters
        :param sequences: A list of sequences to be learned
        :return: Final training loss
        """
        self.criterion = nn.MSELoss()
        # self.criterion = nn.HuberLoss(reduction='mean', delta=0.125)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)

        early_stopping = EarlyStopping(self.model, patience=params.es_patience)

        train_losses = []
        val_losses = []

        self.target_col_indexes = [sequences[0].columns.values.tolist().index(col) for col in
                                   self.model_params.cols_predict] \
            if self.model_params.cols_predict is not None else None

        train_sequences, val_sequences, (scaler, split) = dfs2tensors(sequences,
                                                                      val_perc=val_perc,
                                                                      device=self.model_params.device)

        split.plot_split(title="Train and validation sequences")

        # TODO: Refactor this part to separate functioun
        for epoch in range(params.epochs):
            print(f"Epoch {epoch + 1}/{params.epochs}\n")

            # Forward test data
            train_loss, mae_train_loss = forward_sequences(train_sequences, is_eval=False,
                                                           model=self.model,
                                                           model_params=self.model_params,
                                                           optimizer=self.optimizer,
                                                           criterion=self.criterion,
                                                           target_indexes=self.target_col_indexes)

            # Forward val data
            val_loss, mae_val_loss = forward_sequences(val_sequences, is_eval=True,
                                                       model=self.model,
                                                       model_params=self.model_params,
                                                       optimizer=self.optimizer,
                                                       criterion=self.criterion,
                                                       target_indexes=self.target_col_indexes)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Train loss: {train_loss}, Val loss: {val_loss}")
            print(f"Train MAE: {mae_train_loss}, Val MAE: {mae_val_loss}")

            if early_stopping(val_loss):
                print("Early stopping")
                break

        early_stopping.retrieve()
        self.scaler = scaler

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
        """

        :param sequence_df:
        :param return_inv_transformed: Work on already transformed data
        :return: Return inverse transformed and raw network output.
        """
        sequence_array = sequence_df.values

        sequence_array = self.scaler.transform(sequence_array)

        self.model.eval()

        # Initialize hidden states
        h0 = torch.zeros((self.model_params.n_lstm_layers, self.model.lstm_hidden_size),
                         device=self.model_params.device)
        c0 = torch.zeros((self.model_params.n_lstm_layers, self.model.lstm_hidden_size),
                         device=self.model_params.device)

        out = None

        with torch.no_grad():
            for step in sequence_array:
                step_tensor = torch.Tensor(step).expand((1, -1)).to(self.model_params.device)

                out, (hn, cn) = self.model.forward(step_tensor, h0, c0)

                h0 = hn.detach()
                c0 = cn.detach()

        out = out.to('cpu')

        return self.data_inverse_transform(out) if return_inv_transformed else out
