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

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, batch_first=True, num_layers=3, device=device,
                            dropout=0.2)
        self.fccn = LazyFCCN(hidden_sizes=[64, 32, 16, 8], output_size=output_size, activation=F.selu, device=device,
                             dropout_rate=0.35)

        self.output_size = output_size

        self.called = False

    def forward(self, x, mask):
        if not self.called:
            self.called = True

            with torch.no_grad():
                self.forward(x, mask)

            def weights_init(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

            self.apply(weights_init)

        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)

        # lstm_out = F.dropout(lstm_out, p=0.35)
        lstm_out = lstm_out.reshape(lstm_out.shape[0], -1)
        lstm_out = F.selu(lstm_out)

        output = self.fccn.forward(lstm_out)
        output = F.sigmoid(output)

        return output
