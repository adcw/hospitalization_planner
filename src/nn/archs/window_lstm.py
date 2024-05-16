from typing import Optional, List

from torch import nn
from torch.functional import F

from src.nn.archs.lazy_fccn import LazyFCCN
from src.nn.archs.lazy_mlc import MLConv, ConvLayerData as CLD


class WindowedConvLSTM(nn.Module):
    def __init__(self,
                 n_attr: int,
                 output_size: int,

                 # conv
                 conv_layers_data: Optional[List[CLD]] = None,
                 conv_channel_dropout: float = 0,

                 # LSTM
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

        self.final_activation = final_activation

        self.cldata = conv_layers_data or [
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
        ]

        self.mlconv = MLConv(input_size=n_attr, conv_layers_data=self.cldata, dropout_rate=conv_channel_dropout)

        # self.lstm = nn.LSTM(input_size=self.cldata[-1].channels,
        #                     hidden_size=self.lstm_hidden_size,
        #                     num_layers=self.lstm_layers,
        #                     batch_first=True,
        #                     device=self.device,
        #                     dropout=self.lstm_dropout)

        self.mlp = LazyFCCN(
            hidden_sizes=self.mlp_arch,
            output_size=self.output_size,
            dropout_rate=self.mlp_dropout,
            activation=self.mlp_activation
        )

    def forward(self, x, m):
        # x.shape = (batch, seq, feat)

        x_perm = x.permute(0, 2, 1)
        # (batch, feat, seq)

        x_conv = self.mlconv(x_perm)
        x_conv = x_conv.permute(0, 2, 1)
        # (batch, seq, feat)

        # self.lstm.flatten_parameters()
        # x_lstm, _ = self.lstm(x_conv)
        #
        # mlp_in = x_lstm[:, -1, :]

        output = self.mlp(x_conv.flatten(-2))

        if self.final_activation is not None:
            output = self.final_activation(output)

        return output
