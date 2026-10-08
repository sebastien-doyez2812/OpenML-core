import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .lib.convolutions import DoubleConv

class SegFormerTorch(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, depth=4, initial_filters=64):
        pass

    def forward(self):
        pass

class SegFormer(BaseModel):
    def __init__(self, input_channels=1, output_channels=1, depth=4, initial_filters=64, **kwargs):
        super().__init__(name="SegFormer", **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depth = depth
        self.initial_filters = initial_filters

    def create_model(self):
        self.model = SegFormerTorch(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            depth=self.depth,
            initial_filters=self.initial_filters
        )
        return self.model