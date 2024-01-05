from dataclasses import dataclass
from typing import List, Any

from torch import nn


@dataclass
class ConvLayerData:
    channels: int
    kernel_size: int
    activation: Any
    stride: int = 1


class MLConv(nn.Module):
    def __init__(self,
                 input_size,
                 conv_layers_data: List[ConvLayerData],
                 dropout_rate=0.0):
        super(MLConv, self).__init__()

        layers = []
        in_size = input_size

        for conv_data in conv_layers_data:
            layers.append(
                nn.Conv1d(in_channels=in_size,
                          out_channels=conv_data.channels,
                          kernel_size=conv_data.kernel_size,
                          stride=conv_data.stride,
                          padding='valid'))

            # layers.append(nn.BatchNorm1d(num_features=conv_data.channels))
            if conv_data.activation is not None:
                layers.append(conv_data.activation())

            if dropout_rate != 0:
                layers.append(nn.Dropout(dropout_rate))

            in_size = conv_data.channels

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_layers(x)
