import torch
import torch.nn as nn
import torch.nn.functional as F


class LazyMLP(nn.Module):
    def __init__(self,
                 hidden_sizes,
                 output_size,
                 activation=F.relu,
                 dropout_rate=0.0,
                 device: torch.device = 'cpu'
                 ):
        super(LazyMLP, self).__init__()

        self.input_layer = nn.LazyLinear(hidden_sizes[0], device=device)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], device=device) for i in range(len(hidden_sizes) - 1)
        ])

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.activation(self.input_layer(x))

        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)

        output = self.output_layer(x)
        return output
