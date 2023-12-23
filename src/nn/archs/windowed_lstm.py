from typing import Optional

import torch
from torch import nn
from torch.functional import F


class WindowedLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,

                 # Other
                 device: torch.device = 'cpu',
                 ):
        super().__init__()

        self.test_layer = nn.Linear(in_features=input_size, out_features=16)

    def forward(self, x, masks):
        x = self.test_layer(x)

        return x
