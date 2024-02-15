from typing import Optional

import torch


class EarlyStopping:
    def __init__(self, model: torch.nn.Module, patience: Optional[int] = 5, delta=-0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.model = model

        self.best_state_dict = None

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint()

            return False

        if val_loss >= self.best_score + self.delta:
            self.counter += 1
        else:
            self.best_score = val_loss
            self.save_checkpoint()
            self.counter = 0

        return self.counter >= self.patience

    def save_checkpoint(self):
        self.best_state_dict = self.model.state_dict()

    def retrieve(self):
        """
        Retrieve model state with the lowest val loss
        :return:
        """
        self.model.load_state_dict(self.best_state_dict)
