from typing import Optional, List

from torch import nn
from torch.functional import F

from src.nn.archs.LazyMLC import MLConv, ConvLayerData as CLD
from src.nn.archs.LazyMLP import LazyMLP


class WindowedConvLSTM(nn.Module):
    def __init__(self,
                 n_attr: int,
                 output_size: int,

                 # conv
                 conv_channels: int = 32,
                 conv_kernel_size: int = 5,
                 conv_stride: int = 1,
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

        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout

        self.mlp_arch = mlp_arch or [128, 64, 16]
        self.mpl_dropout = mlp_dropout
        self.mlp_activation = mlp_activation

        self.cldata = conv_layers_data or [
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
        ]

        self.mlconv = MLConv(input_size=n_attr, conv_layers_data=self.cldata, dropout_rate=0)

        self.lstm = nn.LSTM(input_size=self.cldata[-1].channels, hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            batch_first=True,
                            device=self.device, dropout=self.lstm_dropout)

        self.post_lstm_bnm = nn.LazyBatchNorm1d()

        self.mlp = LazyMLP(hidden_sizes=self.mlp_arch,
                           output_size=self.output_size,
                           dropout_rate=self.mpl_dropout,
                           activation=self.mlp_activation
                           )

    def forward(self, x, m):
        # pre_norm
        x_perm = x.permute(0, 2, 1)
        x_conv = self.mlconv(x_perm)
        x_conv = F.selu(x_conv)
        x_conv = x_conv.permute(0, 2, 1)

        self.lstm.flatten_parameters()
        x_lstm, _ = self.lstm(x_conv)
        x_lstm = x_lstm[:, -1, :]
        x_lstm = self.post_lstm_bnm(x_lstm)
        x_lstm = F.selu(x_lstm)

        x_mlp = self.mlp(x_lstm)
        output = F.sigmoid(x_mlp)

        return output
