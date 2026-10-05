import torch
import torch.nn as nn
from .base_model import BaseModel
from .lib.convolutions import DoubleConv
from .lib.attention import AttentionGate

class AttUnetTorch(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, depth=4, initial_filters=64,
                 kernel_size=3, padding=1):
        super(AttUnetTorch, self).__init__()
        self.downs          = nn.ModuleList()
        self.ups            = nn.ModuleList()
        self.attentionGates = nn.ModuleList()
        self.pool           = nn.MaxPool2d(kernel_size=2, stride=2)

        in_c = input_channels
        encoders_output_channels = []
        out_c = initial_filters

        for _ in range(depth):
            self.downs.append(DoubleConv(in_c, out_c, kernel_size=kernel_size, padding=padding))
            in_c = out_c
            out_c *= 2
            encoders_output_channels.append(in_c)

        self.bottleneck = DoubleConv(in_c, out_c, kernel_size=kernel_size, padding=padding)
        
        for enc_c in reversed(encoders_output_channels):
            dec_c = out_c //2
            self.attentionGates.append(AttentionGate(enc_c, dec_c, enc_c//2))
            self.ups.append(nn.ConvTranspose2d(out_c, dec_c, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(out_c, dec_c, kernel_size=kernel_size, padding=padding))

            out_c = dec_c
        self.final_conv = nn.Conv2d(out_c, output_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[-(i // 2 + 1)]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            att_out = self.attentionGates[i // 2] (skip_connection, x)
            x = torch.cat((att_out, x), dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)


class AttentionUNet(BaseModel):
    def __init__(self, input_channels=1, output_channels=1, depth=4, initial_filters=64, **kwargs):
        super().__init__(name="AttentionUNet", **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depth = depth
        self.initial_filters = initial_filters

    def create_model(self):
        self.model = AttUnetTorch(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            depth=self.depth,
            initial_filters=self.initial_filters
        )
        return self.model