import math

from src.nn.callbacks.early_stopping import EarlyStopping


class LrSchedule:
    def __init__(self,
                 optimizer,
                 early_stopping: EarlyStopping,
                 factor=0.8,
                 threshold: float = 0.5,
                 cooldown_factor: float = 0,
                 verbose: int = 0
                 ):
        self.optimizer = optimizer
        self.factor = factor
        self.early_stopping = early_stopping
        self.threshold = threshold
        self.verbose = verbose

        self.cooldown_threshold = math.floor(cooldown_factor * early_stopping.patience)
        self.cooldown_counter = 0

    def step(self):
        es_progress = float(self.early_stopping.counter) / self.early_stopping.patience
        self.verbose > 1 and print(f"Early stopping progress: {es_progress:.3f}")

        if es_progress >= self.threshold and self.cooldown_counter >= self.cooldown_threshold:
            self._update()
            self.cooldown_counter = 0
        elif self.cooldown_counter < self.cooldown_threshold:
            self.cooldown_counter += 1

    def _update(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.factor

        self.early_stopping.retrieve()

        self.verbose > 0 and print(f"Setting lr to {self.optimizer.param_groups[0]['lr']}")
