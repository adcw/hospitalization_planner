import torch
from torch import nn
from torch.nn import functional as F


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
