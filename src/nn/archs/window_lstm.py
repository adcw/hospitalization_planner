from typing import Optional, List

import torch
from torch import nn
from torch.functional import F

from src.nn.archs.lazy_fccn import LazyFCCN
from src.nn.archs.lazy_mlc import MLConv, ConvLayerData as CLD
from src.nn.archs.lazy_mlp import LazyMLP


def masked_min(t: torch.Tensor,
               m: Optional[torch.Tensor],
               dim: int = -1
               ):
    if m is not None:
        masked = t.clone()
        masked[m.bool()] = torch.inf
        t = masked

    result, _ = torch.min(t, dim=dim)
    return result


def masked_max(t: torch.Tensor,
               m: Optional[torch.Tensor],
               dim: int = -1
               ):
    if m is not None:
        masked = t.clone()
        masked[m.bool()] = -torch.inf
        t = masked

    result, _ = torch.max(t, dim=dim)
    return result


def masked_range(t: torch.Tensor,
                 m: torch.Tensor = None,
                 dim: int = -1):
    x_max = masked_max(t, m, dim=dim)
    x_min = masked_min(t, m, dim=dim)
    return abs(x_max - x_min)


class WindowedConvLSTM(nn.Module):
    def __init__(self,
                 n_attr: int,
                 output_size: int,

                 # conv
                 conv_layers_data: Optional[List[CLD]] = None,

                 # LSTM
                 lstm_hidden_size: int = 128,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.2,

                 # MLP
                 mlp_arch: Optional[List] = None,
                 mlp_dropout: float = 0.2,
                 mlp_activation=F.selu,

                 device='cpu'):
        super(WindowedConvLSTM, self).__init__()

        self.device = device

        self.output_size = output_size
        self.n_attr = n_attr

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout

        self.mlp_arch = mlp_arch or [128, 64, 16]
        self.mlp_dropout = mlp_dropout
        self.mlp_activation = mlp_activation

        self.cldata = conv_layers_data or [
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
            # CLD(channels=48, kernel_size=3, activation=nn.Tanh),
            # CLD(channels=32, kernel_size=3, activation=nn.Tanh),
        ]

        self.mlconv = MLConv(input_size=n_attr, conv_layers_data=self.cldata, dropout_rate=0)

        self.lstm = nn.LSTM(input_size=self.cldata[-1].channels,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            batch_first=True,
                            device=self.device, dropout=self.lstm_dropout)

        # self.post_lstm_bnm = nn.LazyBatchNorm1d()

        # self.mlp = LazyMLP(hidden_sizes=self.mlp_arch,
        #                    output_size=self.output_size,
        #                    dropout_rate=self.mpl_dropout,
        #                    activation=self.mlp_activation
        #                    )

        self.mlp = LazyFCCN(
            hidden_sizes=self.mlp_arch,
            output_size=self.output_size,
            dropout_rate=self.mlp_dropout,
            activation=self.mlp_activation
        )

    def forward(self, x, m):
        # x.shape = (batch, seq, feat)

        # (batch, feat)
        # x_range = masked_range(x, m, dim=1)

        # (batch, feat, seq)
        x_perm = x.permute(0, 2, 1)
        x_conv = self.mlconv(x_perm)
        # x_conv = F.selu(x_conv)
        x_conv = x_conv.permute(0, 2, 1)

        self.lstm.flatten_parameters()
        # (batch, seq, feat)
        x_lstm, _ = self.lstm(x_conv)

        # (batch, feat)
        x_lstm = x_lstm[:, -1, :]
        # x_lstm = self.post_lstm_bnm(x_lstm)
        x_lstm = F.tanh(x_lstm)

        # fccn_in = torch.cat([x_lstm, x_range[:, -1:]], dim=1)
        mlp_in = x_lstm

        x_mlp = self.mlp(mlp_in)
        output = F.sigmoid(x_mlp)

        return output
