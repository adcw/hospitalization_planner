from typing import Optional, Dict, Any

import torch


class EarlyStopping:
    def __init__(self, model: torch.nn.Module, patience: Optional[int] = 5, delta=0.0001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.model = model

        self.best_state_dict = None
        self.other_stats = None

    def __call__(self, val_loss, other_stats: Optional[Dict[str, Any]] = None):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(other_stats)

            return False

        if val_loss >= self.best_score - self.delta:
            self.counter += 1
        else:
            self.best_score = val_loss
            self.save_checkpoint(other_stats)
            self.counter = 0

        return self.counter >= self.patience

    def save_checkpoint(self, other_stats: Optional[Dict[str, Any]] = None):
        self.other_stats = other_stats
        self.best_state_dict = self.model.state_dict()

    def retrieve(self):
        """
        Retrieve model state with the lowest val loss
        :return:
        """
        self.model.load_state_dict(self.best_state_dict)
        return self.other_stats
