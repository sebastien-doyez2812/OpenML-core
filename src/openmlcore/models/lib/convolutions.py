import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size = 3, padding = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channel_in , out_channels=channel_out, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        return x