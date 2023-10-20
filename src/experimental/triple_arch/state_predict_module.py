import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from tqdm import tqdm

from src.experimental.triple_arch.archs import StepTimeLSTM
from src.preprocessing.preprocess import normalize_split


def seq2tensors(sequences: list[np.ndarray], device: torch.device):
    tensors = []
    for seq in sequences:
        tensor = torch.Tensor(seq)
        tensor = tensor.to(device)
        tensors.append(tensor)
    return tensors


class StatePredictionModule:
    def __init__(self,
                 n_attr: int,
                 hidden_size: int = 256,

                 device: torch.device = 'cpu'
                 ):

        self.hidden_size = hidden_size
        self.device = device

        self.model = StepTimeLSTM(input_size=n_attr,
                                  hidden_size=self.hidden_size,
                                  output_size=n_attr,
                                  device=device)

        self.model = self.model.to(self.device)

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
                              # desc=f"Training on {split_i + 1}/{n_splits} split",
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
            h0 = torch.randn((1, self.model.hidden_size), device=self.device)
            c0 = torch.randn((1, self.model.hidden_size), device=self.device)

            # Iterate over sequence
            for step_i in range(len(seq) - 1):

                # Get input and output data
                input_step: torch.Tensor = seq[step_i].clone()
                output_step: torch.Tensor = seq[step_i + 1].clone()
                input_step = input_step.expand((1, -1)).to(self.device)
                output_step = output_step.expand((1, -1)).to(self.device)

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

    def evaluate(self,
                 sequences: list[np.ndarray],

                 epochs: int = 5,
                 es_patience: None | int = None,
                 n_splits: int = 2):
        """
        Train the model

        :param sequences: A list of sequences to be learnt
        :param epochs: Number of epochs
        :param es_patience: Early stopping patience
        :param n_splits: The number of split for crossvalidation
        :return:
        """
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Variables to keep track of val loss
        patience_counter = 0 if es_patience is not None else None
        min_val_loss_mean = np.inf

        val_losses = []
        train_losses = []

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}\n")

            kf = KFold(n_splits=n_splits, shuffle=True)

            val_loss_sum = 0
            train_loss_sum = 0

            for split_i, (train_index, val_index) in enumerate(kf.split(sequences)):
                train_sequences = [sequences[i] for i in train_index]
                val_sequences = [sequences[i] for i in val_index]

                train_sequences, val_sequences, _ = normalize_split(train_sequences, val_sequences)
                train_sequences = seq2tensors(train_sequences, self.device)
                val_sequences = seq2tensors(val_sequences, self.device)

                # Train on sequences
                train_loss = self._forward_sequences(train_sequences, is_validation=False)

                # Track validation loss
                val_loss = self._forward_sequences(val_sequences, is_validation=True)

                val_loss_sum += val_loss
                train_loss_sum += train_loss

            val_loss_mean = val_loss_sum / n_splits
            train_loss_mean = train_loss_sum / n_splits

            val_losses.append(val_loss_mean)
            train_losses.append(train_loss_mean)

            print(f"\nMean train loss = {train_loss_mean}, mean val loss = {val_loss_mean}")

            if es_patience is not None:
                if val_loss_mean < min_val_loss_mean:
                    min_val_loss_mean = val_loss_mean
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= es_patience:
                    print("Early stopping")
                    break

        plt.plot(train_losses, label="Train loss")
        plt.plot(val_losses, label="Validation loss")

        plt.legend()
        plt.show()

    def predict(self, sequence: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict next state and return it along with LSTM hidden state.
        The hidden state contains extracted time information about sequence.

        :param sequence:
        :return: Predicted state and hidden lstm state
        """
        self.model.eval()

        # Initialize hidden states
        h0 = torch.randn((1, self.model.hidden_size), device=self.device)
        c0 = torch.randn((1, self.model.hidden_size), device=self.device)

        out = None

        for step in sequence:
            out, (hn, cn) = self.model.forward(step, h0, c0)

            h0 = hn.detach()
            c0 = cn.detach()

        return out, (h0, c0)
