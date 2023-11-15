from typing import Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from tqdm import tqdm

from src.config.config_classes import ModelParams, TrainParams
from src.nn.archs import StepTimeLSTM
from src.preprocessing import normalize_split


def seq2tensors(sequences: list[np.ndarray], device: torch.device):
    tensors = []
    for seq in sequences:
        tensor = torch.Tensor(seq)
        tensor = tensor.to(device)
        tensors.append(tensor)
    return tensors


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
                                  hidden_size=self.model_params.hidden_size,
                                  n_lstm_layers=self.model_params.n_lstm_layers,
                                  output_size=self.n_attr_out * self.model_params.n_steps_predict,
                                  device=self.model_params.device,
                                  fccn_arch=self.model_params.fccn_arch,
                                  fccn_dropout_p=self.model_params.fccn_dropout_p)

        self.model = self.model.to(self.model_params.device)

        self.criterion = None
        self.optimizer = None

        self.scaler = None
        self.target_col_indexes = None

    def _forward_sequences(self,
                           sequences: list[torch.Tensor],
                           is_validation: bool = False,
                           target_indexes: list[int] | None = None
                           ) -> float:
        """
        Forward list of sequences through model.

        :param sequences: list of sequences
        :param is_validation: whether we are validating model instead of training
        :return:
        """
        train_progress = tqdm(sequences, total=sum([len(s) - self.model_params.n_steps_predict for s in sequences]))

        # Track overall loss
        loss_sum = 0

        # Select proper mode
        if is_validation:
            self.model.eval()
        else:
            self.model.train()

        # Forward all sequences
        for _, seq in enumerate(sequences):

            # Clear internal LSTM states
            h0 = torch.randn((self.model_params.n_lstm_layers, self.model.hidden_size), device=self.model_params.device)
            c0 = torch.randn((self.model_params.n_lstm_layers, self.model.hidden_size), device=self.model_params.device)

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

                if not is_validation:
                    self.optimizer.zero_grad()

                if is_validation:
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

                # preserve internal LSTM states
                h0, c0 = hn.detach(), cn.detach()

                # Back-propagation
                if not is_validation:
                    loss.backward()
                    self.optimizer.step()

                train_progress.update(1)

        # Return mean loss
        mean_loss = loss_sum / train_progress.total

        return mean_loss

    def _kfold(self,
               sequences: list[np.ndarray],
               kfold_n_splits: Optional[int] = 5,
               target_indexes: Optional[list[int]] = None
               ):
        kf = KFold(n_splits=kfold_n_splits, shuffle=True)

        val_loss_sum = 0
        train_loss_sum = 0

        for split_i, (train_index, val_index) in enumerate(kf.split(sequences)):
            train_sequences = [sequences[i] for i in train_index]
            val_sequences = [sequences[i] for i in val_index]

            train_sequences, val_sequences, _ = normalize_split(train_sequences, val_sequences)
            train_sequences = seq2tensors(train_sequences, self.model_params.device)
            val_sequences = seq2tensors(val_sequences, self.model_params.device)

            # Train on sequences
            train_loss = self._forward_sequences(train_sequences, is_validation=False, target_indexes=target_indexes)

            # Track validation loss
            val_loss = self._forward_sequences(val_sequences, is_validation=True, target_indexes=target_indexes)

            val_loss_sum += val_loss
            train_loss_sum += train_loss

        val_loss_mean = val_loss_sum / kfold_n_splits
        train_loss_mean = train_loss_sum / kfold_n_splits

        return train_loss_mean, val_loss_mean

    def train(self, params: TrainParams, sequences: list[pd.DataFrame]):
        """
        Train the model

        :param params: Training parameters
        :param sequences: A list of sequences to be learned
        :return: Scaler if training is performed, otherwise None
        """
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        patience_counter = 0 if params.es_patience is not None else None
        min_target_loss_mean = np.inf
        val_losses = []
        train_losses = []

        self.target_col_indexes = [sequences[0].columns.values.tolist().index(col) for col in
                                   self.model_params.cols_predict] \
            if self.model_params.cols_predict is not None else None

        sequences = [s.values for s in sequences]

        train_sequences, _, scaler = normalize_split(sequences, None)
        train_sequences = seq2tensors(train_sequences, self.model_params.device)

        for epoch in range(params.epochs):
            print(f"Epoch {epoch + 1}/{params.epochs}\n")

            curr_target_loss_mean = self._forward_sequences(train_sequences, is_validation=False,
                                                            target_indexes=self.target_col_indexes)
            train_loss_mean = curr_target_loss_mean

            val_losses.append(curr_target_loss_mean)
            train_losses.append(train_loss_mean)

            print(f"\nMean train loss = {train_loss_mean}")
            print(f"mean val loss = {curr_target_loss_mean}")

            if params.es_patience is not None:
                if curr_target_loss_mean < min_target_loss_mean:
                    min_target_loss_mean = curr_target_loss_mean
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= params.es_patience:
                    print("Early stopping")
                    break

        self.scaler = scaler

        plt.plot(train_losses, label="Train loss")
        plt.plot(val_losses, label="Validation loss")
        plt.legend()
        plt.show()

    def eval(self, params: TrainParams, sequences: list[pd.DataFrame]):
        """
        Evaluate the model

        :param params: Evaluation parameters
        :param sequences: A list of sequences to be evaluated
        """
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        patience_counter = 0 if params.es_patience is not None else None
        min_target_loss_mean = np.inf
        val_losses = []
        target_col_indexes = [sequences[0].columns.values.tolist().index(col) for col in
                              self.model_params.cols_predict] \
            if self.model_params.cols_predict is not None else None

        sequences = [s.values for s in sequences]

        for epoch in range(params.epochs):
            print(f"Epoch {epoch + 1}/{params.epochs}\n")
            train_loss_mean, curr_target_loss_mean = self._kfold(sequences=sequences,
                                                                 kfold_n_splits=params.eval_n_splits,
                                                                 target_indexes=target_col_indexes)

            val_losses.append(curr_target_loss_mean)
            print(f"\nMean train loss = {train_loss_mean}, mean val loss = {curr_target_loss_mean}")

            if params.es_patience is not None:
                if curr_target_loss_mean < min_target_loss_mean:
                    min_target_loss_mean = curr_target_loss_mean
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= params.es_patience:
                    print("Early stopping")
                    break

        plt.plot(val_losses, label="Validation loss")
        plt.legend()
        plt.show()

    def predict(self, sequence_df: pd.DataFrame):
        sequence_array = sequence_df.values

        sequence_array = self.scaler.transform(sequence_array)

        self.model.eval()

        # Initialize hidden states
        h0 = torch.randn((self.model_params.n_lstm_layers, self.model.hidden_size), device=self.model_params.device)
        c0 = torch.randn((self.model_params.n_lstm_layers, self.model.hidden_size), device=self.model_params.device)

        out = None

        with torch.no_grad():
            for step in sequence_array:
                step_tensor = torch.Tensor(step).expand((1, -1)).to(self.model_params.device)

                out, (hn, cn) = self.model.forward(step_tensor, h0, c0)

                h0 = hn.detach()
                c0 = cn.detach()

        out = out.to('cpu')
        scale_ = self.scaler.scale_[self.target_col_indexes]
        min_ = self.scaler.min_[self.target_col_indexes]
        out = out * scale_ + min_

        return out

    # def _predict_raw(self, sequence: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    #     """
    #     Predict next state and return it along with LSTM hidden state.
    #     The hidden state contains extracted time information about sequence_df.
    #
    #     :param sequence:
    #     :return: Predicted state and hidden lstm state
    #     """
    #     self.model.eval()
    #
    #     # Initialize hidden states
    #     h0 = torch.randn((self.model_params.n_lstm_layers, self.model.hidden_size), device=self.model_params.device)
    #     c0 = torch.randn((self.model_params.n_lstm_layers, self.model.hidden_size), device=self.model_params.device)
    #
    #     out = None
    #
    #     with torch.no_grad():
    #         for step in sequence:
    #             out, (hn, cn) = self.model.forward(step, h0, c0)
    #
    #             h0 = hn.detach()
    #             c0 = cn.detach()
    #
    #     return out, (h0, c0)
