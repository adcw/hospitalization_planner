import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from src.experimental.triple_arch.archs import StepTimeLSTM
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

torch.autograd.set_detect_anomaly(True)


class StatePredictionModule:
    def __init__(self,
                 n_attr: int,
                 hidden_size: int = 256,

                 device: torch.device = 'cpu'
                 ):

        self.hidden_size = hidden_size
        self.device = device

        self.model = StepTimeLSTM(input_size=n_attr, hidden_size=256, output_size=n_attr, device=device)

        self.is_trained = False

        self._last_mean_vloss = np.inf
        self._min_mean_vloss = np.inf
        self._patience_counter = 0

        pass

    def _forward_sequences(self, progress: tqdm,
                           is_validation: bool = False):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001) if not is_validation else None
        validation_loss_sum = 0
        last_loss = None

        seq_i = 1
        step_i = 1
        for seq_i, seq in progress:

            # clear internal LSTM states
            h0 = torch.randn((1, self.model.hidden_size), device=self.device)
            c0 = torch.randn((1, self.model.hidden_size), device=self.device)

            # iterate over sequence
            for step_i in range(len(seq) - 1):
                # get input and output data
                input_step: torch.Tensor = seq[step_i].clone()
                output_step: torch.Tensor = seq[step_i + 1].clone()
                input_step = input_step.expand((1, -1)).to(self.device)
                output_step = output_step.expand((1, -1)).to(self.device)

                if not is_validation:
                    optimizer.zero_grad()

                if is_validation:
                    with torch.no_grad():
                        outputs, (hn, cn) = self.model(input_step, h0, c0)
                else:
                    outputs, (hn, cn) = self.model(input_step, h0, c0)

                loss = criterion(outputs, output_step)
                last_loss = loss.item()
                progress.set_postfix({"Loss": last_loss, "Seq indx": seq_i})
                validation_loss_sum += last_loss

                # preserve internal LSTM states
                h0, c0 = hn.detach(), cn.detach()

                if not is_validation:
                    loss.backward()
                    optimizer.step()

        mean_loss = validation_loss_sum / ((seq_i + 1) * (step_i + 1))
        print(f"\nMean loss: {mean_loss:.6f}, {last_loss=}")

        if is_validation:
            self._last_mean_vloss = mean_loss

    def train(self,
              sequences: list[torch.Tensor],

              epochs: int = 5,
              es_patience: None | int = None,
              n_splits: int = 2):

        if es_patience is not None:
            self._patience_counter = 0

        for epoch in range(epochs):
            kf = KFold(n_splits=n_splits, shuffle=True)

            for train_index, val_index in kf.split(sequences):
                train_sequences = [sequences[i] for i in train_index]
                val_sequences = [sequences[i] for i in val_index]

                # Train on sequences
                progress = tqdm(enumerate(train_sequences), desc=f"Training on {epoch=}")
                self._forward_sequences(progress, is_validation=False)

                # Track validation loss
                progress = tqdm(enumerate(val_sequences), desc=f"Validating on {epoch=}")
                self._forward_sequences(progress, is_validation=True)

            if es_patience is not None:
                if self._last_mean_vloss < self._min_mean_vloss:
                    self._min_mean_vloss = self._last_mean_vloss
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1

                if self._patience_counter >= es_patience:
                    print("Early stopping")
                    break

        print(f"{self._last_mean_vloss=:.6f}")
        self.is_trained = True
