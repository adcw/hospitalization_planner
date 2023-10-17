import torch
from torch import nn
from torch.nn import functional as F


class StepTimeLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: torch.device = 'cpu'):
        super(StepTimeLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, device=device)
        self.fccn = FCCN(input_size=hidden_size, hidden_sizes=[32] * 5, output_size=output_size, dropout_rate=0.15,
                         activation=F.elu, device=device)

    def forward(self, x, h0, c0):
        """
        :param x: Input row
        :param h0: Previous h0 hidden state
        :param c0: Previous c0 hidden state
        :return: Predicted row and hidden states (hn, cn)
        """

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fccn(out)
        out = F.sigmoid(out)
        return out, (hn, cn)


class FCCN(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_sizes: list[int],
                 output_size: int,
                 dropout_rate: float = 0.1,

                 activation=F.tanh,
                 device: torch.device = 'cpu'
                 ):
        super(FCCN, self).__init__()

        self.dropout_rate = dropout_rate
        self.activation = activation
        self.device = device

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0], device=self.device)

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
