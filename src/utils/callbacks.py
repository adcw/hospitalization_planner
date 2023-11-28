import torch


class EarlyStopping:
    def __init__(self, model, patience=5, delta=0, path=None):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.model = model

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint()

            return False

        if val_loss > self.best_score + self.delta:
            self.counter += 1
        else:
            self.best_score = val_loss
            self.save_checkpoint()
            self.counter = 0

        return self.counter >= self.patience

    def save_checkpoint(self):
        if self.path is not None:
            torch.save(self.model.state_dict(), self.path)
