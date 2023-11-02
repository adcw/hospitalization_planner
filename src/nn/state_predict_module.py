from typing import Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from tqdm import tqdm

from src.nn.archs import StepTimeLSTM
from src.config.net_params import NetParams, TrainParams
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
                 net_params: NetParams,
                 n_steps_predict: int = 1,
                 cols_predict: Optional[list[str]] = None,
                 ):

        self.net_params = net_params

        self.model = StepTimeLSTM(input_size=self.net_params.n_attr,
                                  hidden_size=self.net_params.hidden_size,
                                  lstm_num_layers=self.net_params.n_lstm_layers,
                                  output_size=self.net_params.n_attr,
                                  device=self.net_params.device)

        self.model = self.model.to(self.net_params.device)

        self.criterion = None
        self.optimizer = None

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def _forward_sequences(self,
                           sequences: list[torch.Tensor],
                           is_validation: bool = False) -> float:
        """
        Forward list of sequences through model.

        :param sequences: list of sequences
        :param is_validation: whether we are validating model instead of training
        :return:
        """
        train_progress = tqdm(sequences,
                              # desc=f"Training on {split_i + 1}/{kfold_n_splits} split",
                              total=sum([len(s) - 1 for s in sequences]))

        # Track overall loss
        loss_sum = 0

        # Select proper mode
        if is_validation:
            self.model.eval()
        else:
            self.model.train()

        # Forward all sequences
        for seq_i, seq in enumerate(sequences):

            # Clear internal LSTM states
            h0 = torch.randn((self.net_params.n_lstm_layers, self.model.hidden_size), device=self.net_params.device)
            c0 = torch.randn((self.net_params.n_lstm_layers, self.model.hidden_size), device=self.net_params.device)

            # Iterate over sequence
            for step_i in range(len(seq) - 1):

                # Get input and output data
                input_step: torch.Tensor = seq[step_i].clone()
                output_step: torch.Tensor = seq[step_i + 1].clone()
                input_step = input_step.expand((1, -1)).to(self.net_params.device)
                output_step = output_step.expand((1, -1)).to(self.net_params.device)

                if not is_validation:
                    self.optimizer.zero_grad()

                if is_validation:
                    with torch.no_grad():
                        outputs, (hn, cn) = self.model(input_step, h0, c0)
                else:
                    outputs, (hn, cn) = self.model(input_step, h0, c0)

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
               ):
        kf = KFold(n_splits=kfold_n_splits, shuffle=True)

        val_loss_sum = 0
        train_loss_sum = 0

        for split_i, (train_index, val_index) in enumerate(kf.split(sequences)):
            train_sequences = [sequences[i] for i in train_index]
            val_sequences = [sequences[i] for i in val_index]

            train_sequences, val_sequences, _ = normalize_split(train_sequences, val_sequences)
            train_sequences = seq2tensors(train_sequences, self.net_params.device)
            val_sequences = seq2tensors(val_sequences, self.net_params.device)

            # Train on sequences
            train_loss = self._forward_sequences(train_sequences, is_validation=False)

            # Track validation loss
            val_loss = self._forward_sequences(val_sequences, is_validation=True)

            val_loss_sum += val_loss
            train_loss_sum += train_loss

        val_loss_mean = val_loss_sum / kfold_n_splits
        train_loss_mean = train_loss_sum / kfold_n_splits

        return train_loss_mean, val_loss_mean

    def train(self,
              sequences: list[np.ndarray],
              params: TrainParams,
              mode: Literal['train', 'eval'] = 'train'
              ):
        """
        Train the model

        :param params:
        :param mode:
        :param sequences: A list of sequences to be learnt
        :return: Return scaler if there is no kfold_n_splits argument, otherwise return None.
        """
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Variables to keep track of val loss
        patience_counter = 0 if params.es_patience is not None else None

        min_target_loss_mean = np.inf

        val_losses = []
        train_losses = []
        scaler = None

        # Run epochs
        for epoch in range(params.epochs):
            print(f"Epoch {epoch + 1}/{params.epochs}\n")

            # We perform KFold evaluation
            if mode == 'eval':
                train_loss_mean, curr_target_loss_mean = self._kfold(sequences=sequences,
                                                                     kfold_n_splits=params.eval_n_splits)

            # We perform regular training
            else:

                train_sequences, _, scaler = normalize_split(sequences, None)
                train_sequences = seq2tensors(train_sequences, self.net_params.device)

                curr_target_loss_mean = self._forward_sequences(train_sequences, is_validation=False)
                train_loss_mean = curr_target_loss_mean

            val_losses.append(curr_target_loss_mean)
            train_losses.append(train_loss_mean)

            print(
                f"\nMean train loss = {train_loss_mean}, "
                f"mean val loss = {curr_target_loss_mean if mode == 'eval' else '-'}")

            if params.es_patience is not None:
                if curr_target_loss_mean < min_target_loss_mean:
                    min_target_loss_mean = curr_target_loss_mean
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= params.es_patience:
                    print("Early stopping")
                    break

        plt.plot(train_losses, label="Train loss")
        plt.plot(val_losses, label="Validation loss")

        plt.legend()
        plt.show()

        return scaler

    def predict(self):

        pass

    def predict_raw(self, sequence: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict next state and return it along with LSTM hidden state.
        The hidden state contains extracted time information about sequence.

        :param sequence:
        :return: Predicted state and hidden lstm state
        """
        self.model.eval()

        # Initialize hidden states
        h0 = torch.randn((self.net_params.n_lstm_layers, self.model.hidden_size), device=self.net_params.device)
        c0 = torch.randn((self.net_params.n_lstm_layers, self.model.hidden_size), device=self.net_params.device)

        out = None

        with torch.no_grad():
            for step in sequence:
                out, (hn, cn) = self.model.forward(step, h0, c0)

                h0 = hn.detach()
                c0 = cn.detach()

        return out, (h0, c0)
