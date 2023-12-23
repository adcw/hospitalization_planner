from typing import Optional, Tuple

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from src.config.dataclassess import ModelParams, TrainParams
from src.models.utils import dfs2tensors
from src.models.windowed_forward import windowed_forward
from src.nn.archs.windowed_lstm import WindowedLSTM
from src.nn.callbacks.early_stopping import EarlyStopping


class WindowedModule:
    def __init__(self,
                 params: ModelParams,
                 n_attr_in: int
                 ):
        self.n_attr_in = n_attr_in
        self.n_attr_out = len(params.cols_predict) if params.cols_predict is not None else n_attr_in
        self.model_params = params

        self.model = WindowedLSTM(input_size=self.n_attr_in,
                                  output_size=self.n_attr_out * self.model_params.n_steps_predict,
                                  device=self.model_params.device)

        self.model = self.model.to(self.model_params.device)

        self.criterion = None
        self.optimizer = None

        self.scaler: Optional[MinMaxScaler] = None
        self.target_col_indexes = None

    def train(self, params: TrainParams, sequences: list[pd.DataFrame], plot: bool = True,
              val_perc: float = 0.2) -> Tuple[float, float]:
        self.criterion = nn.MSELoss()
        # self.criterion = nn.HuberLoss(reduction='mean', delta=0.125)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        early_stopping = EarlyStopping(self.model, patience=params.es_patience)

        train_losses = []
        val_losses = []

        self.target_col_indexes = [sequences[0].columns.values.tolist().index(col) for col in
                                   self.model_params.cols_predict] \
            if self.model_params.cols_predict is not None else None

        train_sequences, val_sequences, self.scaler = dfs2tensors(sequences, val_perc=val_perc,
                                                                  limit=params.sequence_limit,
                                                                  device=self.model_params.device)

        for epoch in range(params.epochs):
            print(f"Epoch {epoch + 1}/{params.epochs}\n")

            train_loss, mae_train_loss = windowed_forward(train_sequences, is_eval=False,
                                                          model=self.model,
                                                          model_params=self.model_params,
                                                          optimizer=self.optimizer,
                                                          criterion=self.criterion,
                                                          target_indexes=self.target_col_indexes)

            pass

            val_loss, mae_val_loss = ...

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
