from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

import warnings

warnings.filterwarnings("ignore",
                        message="'Lazy modules are a new feature under heavy development '\
                      'so changes to the API or functionality can happen at any moment.'")


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
        self.memory_arranger = LazyFCCN(hidden_sizes=[memory_size * 8, memory_size * 4, memory_size * 2],
                                        output_size=memory_size,
                                        dropout_rate=self.lstm_dropout,
                                        activation=torch.tanh,
                                        device=device)

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
        lstm_output = torch.cat([lstm_output, hn, cn])

        lstm_output = lstm_output.view(1, -1)
        out = self.output_interpreter(lstm_output)

        lstm_memory = torch.cat([h0, c0, hn, cn])
        lstm_memory = lstm_memory.view(1, -1)
        lstm_memory_out = self.memory_arranger(lstm_memory)
        lstm_memory_out = lstm_memory_out.view(2 * self.lstm_num_layers, self.lstm_hidden_size)

        hn, cn = lstm_memory_out.split(self.lstm_num_layers)

        return out, (hn, cn)


class LazyFCCN(nn.Module):
    def __init__(self,
                 hidden_sizes: list[int],
                 output_size: int,
                 dropout_rate: float = 0.1,

                 activation=F.sigmoid,
                 device: torch.device = 'cpu'
                 ):
        super(LazyFCCN, self).__init__()

        self.dropout_rate = dropout_rate
        self.activation = activation
        self.device = device

        # Input layer
        self.input_layer = nn.LazyLinear(hidden_sizes[0], device=self.device)

        # Hidden layers with dropout
        # We are using Lazy Modules, that way the network calculates layer input shapes on its own,
        # what is crucial because of the connection complexity
        hidden_layers = []
        for i in range(len(hidden_sizes) - 1):
            hidden_layers.append(nn.LazyLinear(hidden_sizes[i + 1], device=self.device))

        self.hidden_layers = nn.ModuleList(hidden_layers)

        # Output layer
        self.output_layer = nn.LazyLinear(output_size, device=self.device)

    def forward(self, x):
        last_output = self.activation(self.input_layer(x))
        joined_outputs = torch.cat([x, last_output], dim=1)

        for layer in self.hidden_layers:
            # joining current layer output with network input forming cascade
            output = self.activation(layer(joined_outputs))
            last_output = F.dropout(output, p=self.dropout_rate)
            joined_outputs = torch.cat([joined_outputs, last_output], dim=1)

        # apply last layer and activate with sigmoid to squash output between 0 and 1
        final_output = self.output_layer(joined_outputs)

        return final_output
