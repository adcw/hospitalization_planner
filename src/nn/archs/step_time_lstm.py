from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.functional import F

from src.nn.archs.lazy_fccn import LazyFCCN
from src.nn.layers.attention import LazySelfAttention


class StepTimeLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,

                 # LSTM params
                 n_lstm_layers: int = 2,
                 lstm_hidden_size: int = 128,
                 lstm_dropout: float = 0.2,

                 # FCCN params
                 fccn_arch: list[int] | None = None,
                 fccn_dropout_p: float = 0.15,
                 fccn_activation=F.relu,

                 # Other
                 device: torch.device = 'cpu',
                 ):
        super(StepTimeLSTM, self).__init__()

        fccn_arch = [32] * 5 if fccn_arch is None else fccn_arch
        self.fccn_arch = fccn_arch

        self.fccn_dropout_p = fccn_dropout_p
        self.fccn_activation = fccn_activation
        self.lstm_num_layers = n_lstm_layers
        self.lstm_dropout = lstm_dropout

        self.lstm_hidden_size = lstm_hidden_size
        self.output_size = output_size

        self.device = device

        self.lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True, device=device,
                            num_layers=self.lstm_num_layers, dropout=self.lstm_dropout)

        self.output_interpreter = LazyFCCN(hidden_sizes=self.fccn_arch,
                                           output_size=self.output_size,
                                           dropout_rate=self.fccn_dropout_p,
                                           activation=self.fccn_activation,
                                           device=device)

        memory_size = self.lstm_hidden_size * self.lstm_num_layers * 2

        # self.memory_arranger = LazyFCCN(hidden_sizes=[memory_size * 4, memory_size * 2],
        #                                 output_size=memory_size,
        #                                 dropout_rate=self.lstm_dropout,
        #                                 activation=torch.tanh,
        #                                 device=device)

        self.memory_arranger = LazyFCCN(hidden_sizes=[memory_size * 8, memory_size * 4, memory_size * 2],
                                        output_size=memory_size,
                                        dropout_rate=self.lstm_dropout,
                                        activation=torch.tanh,
                                        device=device)

        self.attention = None

    def forward(self, x, h0: Optional[torch.Tensor] = None, c0: Optional[torch.Tensor] = None):
        """
        :param x: Input row
        :param h0: Previous h0 hidden state
        :param c0: Previous c0 hidden state
        :return: Predicted row and hidden states (hn, cn)
        """

        if h0 is None:
            h0 = torch.zeros((self.lstm_num_layers, self.lstm_hidden_size),
                             device=self.device)

        if c0 is None:
            c0 = torch.zeros((self.lstm_num_layers, self.lstm_hidden_size),
                             device=self.device)

        self.lstm.flatten_parameters()

        lstm_output, (hn, cn) = self.lstm(x, (h0, c0))

        lstm_output = lstm_output.view(1, -1)
        out = self.output_interpreter(lstm_output)
        out = F.sigmoid(out)

        lstm_memory = torch.cat([h0, c0, hn, cn])
        lstm_memory = lstm_memory.view(1, -1)

        # if self.attention is None:
        #     self.attention = torch.nn.MultiheadAttention(lstm_memory.shape[1], 1, device=lstm_memory.device)
        #
        # attended, _ = self.attention(lstm_memory, lstm_memory, lstm_memory)
        #
        # lstm_memory = torch.cat([attended, lstm_memory], dim=1)

        lstm_memory_out = self.memory_arranger(lstm_memory)
        lstm_memory_out = F.tanh(lstm_memory_out)

        lstm_memory_out = lstm_memory_out.view(2 * self.lstm_num_layers, self.lstm_hidden_size)

        hn, cn = lstm_memory_out.split(self.lstm_num_layers)

        # hn_flat, cn_flat = lstm_memory.split(h0.shape[0] * h0.shape[1], dim=1)[-2:]
        # hn = hn_flat.reshape(hn.shape)
        # cn = cn_flat.reshape(cn.shape)

        return out, (hn, cn)
