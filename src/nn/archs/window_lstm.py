from typing import Optional, List

import torch
from torch import nn
from torch.functional import F

from src.nn.archs.lazy_fccn import LazyFCCN
from src.nn.archs.lazy_mlc import MLConv, ConvLayerData as CLD


class WindowedConvLSTM(nn.Module):
    def __init__(self,
                 n_attr: int,
                 output_size: int,

                 # conv
                 with_conv: bool = True,
                 conv_layers_data: Optional[List[CLD]] = None,
                 conv_channel_dropout: float = 0,

                 # LSTM
                 with_lstm: bool = False,
                 lstm_hidden_size: int = 128,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.2,

                 # MLP
                 mlp_arch: Optional[List] = None,
                 mlp_dropout: float = 0.2,
                 mlp_activation=F.selu,

                 final_activation=F.sigmoid,

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

        self.with_lstm = with_lstm
        self.with_conv = with_conv

        self.final_activation = final_activation

        self.cldata = conv_layers_data or [
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
        ]

        if self.with_conv:
            self.mlconv = MLConv(input_size=n_attr, conv_layers_data=self.cldata, dropout_rate=conv_channel_dropout)
        else:
            self.mlconv = None

        if self.with_lstm:
            self.lstm = nn.LSTM(input_size=self.cldata[-1].channels if self.with_conv else n_attr,
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.lstm_layers,
                                batch_first=True,
                                device=self.device,
                                dropout=self.lstm_dropout)
        else:
            self.lstm = None

        self.mlp = LazyFCCN(
            hidden_sizes=self.mlp_arch,
            output_size=self.output_size,
            dropout_rate=self.mlp_dropout,
            activation=self.mlp_activation
        )

    def forward(self, x, x_c):
        # x.shape = (batch, seq, feat)

        if self.mlconv is not None:
            x_perm = x.permute(0, 2, 1)
            # (batch, feat, seq)

            if x_c is not None:
                x_c_2 = torch.reshape(x_c, (x_c.shape[0], x_c.shape[1], 1))
                x_c_2 = x_c_2.repeat(1, 1, x.shape[1])
                x_perm = torch.concat((x_perm, x_c_2), dim=1)

            x_conv = self.mlconv(x_perm)
            x = x_conv.permute(0, 2, 1)

        if self.lstm is not None:
            self.lstm.flatten_parameters()
            x_lstm, _ = self.lstm(x)
            x = x_lstm[:, -1, :]

        if x.ndim > 2:
            x = x.flatten(-2)

        output = self.mlp(x)

        if self.final_activation is not None:
            output = self.final_activation(output)

        return output


if __name__ == '__main__':
    # Ustalanie parametrów modelu
    n_attr = 10
    output_size = 6
    batch_size = 4
    seq_length = 20

    cldata = [
        CLD(channels=64, kernel_size=3, activation=nn.SELU),
        CLD(channels=128, kernel_size=3, activation=nn.SELU),
        # CLD(channels=256, kernel_size=3, activation=nn.SELU),
        # CLD(channels=512, kernel_size=3, activation=nn.SELU),
    ]

    # Tworzenie instancji modelu
    model = WindowedConvLSTM(
        n_attr=n_attr,
        output_size=output_size,
        device='cpu',

        with_conv=False,
        with_lstm=False,

        conv_layers_data=cldata
    )

    # Generowanie losowego tensora wejściowego
    x = torch.randn(batch_size, seq_length, n_attr)
    # x_c = torch.randn(batch_size, n_attr)

    # Przepuszczanie tensora przez model
    output = model(x, None)

    pass
