from typing import Optional

import torch
from torch import nn
from torch.functional import F

from src.nn.archs.lazy_fccn import LazyFCCN
from src.nn.layers.attention import SelfAttentionLayer


class WindowedLSTM(nn.Module):
    def __init__(self, input_size, output_size, device='cpu'):
        super(WindowedLSTM, self).__init__()
        self.device = device

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, batch_first=True, num_layers=2, device=device)
        self.fccn = LazyFCCN(hidden_sizes=[128, 64, 64, 32], output_size=output_size, activation=F.tanh, device=device)

    def forward(self, x, mask):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.reshape(lstm_out.shape[0], -1)
        lstm_out = F.tanh(lstm_out)

        output = self.fccn.forward(lstm_out)
        output = F.sigmoid(output)

        return output
