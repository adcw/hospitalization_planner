from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from tqdm import tqdm

from src.config.parsing import ModelParams, TrainParams
from src.models.utils import dfs2tensors
from src.nn.archs.step_time_lstm import StepTimeLSTM

from src.nn.callbacks.early_stopping import EarlyStopping
from src.nn.callbacks.metrics import MAECounter


class StatePredictionModule:
    def __init__(self,
                 params: ModelParams,
                 n_attr_in: int
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

    def _forward_sequences(self,
                           sequences: list[torch.Tensor],
                           is_eval: bool = False,
                           target_indexes: list[int] | None = None
                           ) -> Tuple[float, float]:
        """
        Forward list of sequences through model.

        :param sequences: list of sequences
        :param is_eval: whether we are evaluating model instead of training
        :return:
        """
        train_progress = tqdm(sequences, total=sum([len(s) - self.model_params.n_steps_predict for s in sequences]))

        # Track overall loss
        loss_sum = 0
        mae_counter = MAECounter()

        # Select proper mode
        if is_eval:
            self.model.eval()
        else:
            self.model.train()

        # Forward all sequences
        for _, seq in enumerate(sequences):
            h0 = None
            c0 = None

            # Iterate over sequence_df
            for step_i in range(len(seq) - self.model_params.n_steps_predict):

                # Get input and output data
                input_step: torch.Tensor = seq[step_i].clone()
                output_step: torch.Tensor = seq[step_i + 1:step_i + 1 + self.model_params.n_steps_predict].clone()

                if target_indexes is not None:
                    output_step = output_step[:, target_indexes]

                input_step = input_step.expand((1, -1)).to(self.model_params.device)

                if self.model_params.n_steps_predict == 1:
                    output_step = output_step.expand((1, -1)).to(self.model_params.device)

                if not is_eval:
                    self.optimizer.zero_grad()

                if is_eval:
                    with torch.no_grad():
                        outputs, (hn, cn) = self.model(input_step, h0, c0)
                else:
                    outputs, (hn, cn) = self.model(input_step, h0, c0)

                outputs = outputs.view(self.model_params.n_steps_predict,
                                       round(outputs.shape[1] / self.model_params.n_steps_predict))

                # Calculate losses
                loss = self.criterion(outputs, output_step)
                last_loss = loss.item()
                train_progress.set_postfix({"Loss": last_loss})
                loss_sum += last_loss

                mae_counter.publish(outputs, output_step)

                # preserve internal LSTM states
                h0, c0 = hn.detach(), cn.detach()

                # Back-propagation
                if not is_eval:
                    loss.backward()
                    self.optimizer.step()

                train_progress.update(1)

        # Return mean loss
        mean_loss = loss_sum / train_progress.total
        mean_mae_loss = mae_counter.retrieve()

        return mean_loss, mean_mae_loss

    def train(self, params: TrainParams, sequences: list[pd.DataFrame], plot: bool = True,
              val_perc: float = 0.2) -> Tuple[float, float]:
        """
        Train the model

        :param plot:
        :param params: Training parameters
        :param sequences: A list of sequences to be learned
        :return: Final training loss
        """
        # self.criterion = nn.MSELoss()
        self.criterion = nn.HuberLoss(reduction='mean', delta=0.125)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9,
                                   nesterov=True)

        early_stopping = EarlyStopping(self.model, patience=params.es_patience)

        train_losses = []
        val_losses = []

        self.target_col_indexes = [sequences[0].columns.values.tolist().index(col) for col in
                                   self.model_params.cols_predict] \
            if self.model_params.cols_predict is not None else None

        train_sequences, val_sequences, scaler = dfs2tensors(sequences, val_perc=val_perc, limit=params.sequence_limit,
                                                             device=self.model_params.device)

        # TODO: Refactor this part to separate functioun
        for epoch in range(params.epochs):
            print(f"Epoch {epoch + 1}/{params.epochs}\n")

            # Forward test data
            train_loss, mae_train_loss = self._forward_sequences(train_sequences, is_eval=False,
                                                                 target_indexes=self.target_col_indexes)

            # Forward val data
            val_loss, mae_val_loss = self._forward_sequences(val_sequences, is_eval=True,
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
