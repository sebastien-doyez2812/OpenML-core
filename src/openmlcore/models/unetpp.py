import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .lib.convolutions import DoubleConv

class UnetPPTorch(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, depth=4, initial_filters=64,
                 kernel_size=3, padding=1):
        super(UnetPPTorch, self).__init__()
        self.depth = depth
        self.downs      = nn.ModuleList()
        self.inter_down = nn.ModuleList()

        filters = [initial_filters * (2**i) for i in range(self.depth + 1)]
        self.pool = nn.MaxPool2d(kernel_size=2, stride= 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.convs = nn.ModuleDict()

        for i in range(self.depth + 1):
            for j in range (self.depth + 1 -i):
                if j == 0:
                    in_c = input_channels if i == 0 else filters[i-1]
                    out_c = filters[i]

                else:
                    in_c = j * filters[i] + filters[i + 1]
                    out_c = filters[i]

                self.convs[f"conv_{i}_{j}"] = DoubleConv(in_c, out_c, kernel_size=kernel_size, padding=padding)

        self.final_conv = DoubleConv(filters[0], output_channels, kernel_size=kernel_size, padding=padding)


    def forward(self, x):
        x_node = {}
        x_node["0_0"] = self.convs["conv_0_0"](x)

        # All the X_node on the left, the one called X^[i, 0]:
        for i in range(1, self.depth + 1):
            x_down = self.pool(x_node[f"{i-1}_0"])
            x_node[f"{i}_0"] = self.convs[f"conv_{i}_0"](x_down)

        # Loop for X^[i, j]
        for j in range(1, self.depth +1):
            for i in range(self.depth + 1 -j):
                x_up = self.up(x_node[f"{i+1}_{j-1}"])

                if x_up.shape[2:] != x_node[f'{i}_0'].shape[2:]:
                    x_up = F.interpolate(
                        x_up, size=x_node[f'{i}_0'].shape[2:], 
                        mode="bilinear", align_corners=True
                    )
                
                skip_concat = [x_node[f'{i}_{k}'] for k in range(j)]
                skip_concat.append(x_up)
                
                concat_features = torch.cat(skip_concat, dim=1)
                x_node[f'{i}_{j}'] = self.convs[f'conv_{i}_{j}'](concat_features)


        # TODO: implementation of Deep supervision:
        # if self.deepsupervision:
        #     # TODO
        # else:

        return self.final_conv(x_node[f"0_{self.depth}"])

    
class UNetPP(BaseModel):
    def __init__(self, input_channels=1, output_channels=1, depth=4, initial_filters=64, **kwargs):
        super().__init__(name="UNet++", **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depth = depth
        self.initial_filters = initial_filters

    def create_model(self):
        self.model = UnetPPTorch(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            depth=self.depth,
            initial_filters=self.initial_filters
        )
        return self.model