import torch


class MAECounter:
    def __init__(self):
        self.mae_summed = 0
        self.counter = 0

    def publish(self, true: torch.Tensor, pred: torch.Tensor):
        with torch.no_grad():
            mae = torch.mean(abs(true - pred))

        self.counter += 1
        self.mae_summed += mae

    def retrieve(self):
        if self.counter == 0:
            raise RuntimeWarning("Called retrieve but there is no data published")
        return (self.mae_summed / self.counter) if self.counter > 0 else None

    def clear(self):
        self.counter = 0
        self.mae_summed = 0
