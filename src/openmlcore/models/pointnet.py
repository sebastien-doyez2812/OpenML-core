from .lib.tnet import Tnet
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class PointNetTorch(nn.Module):
    def __init__(self, input_dims= 3, output_dims=13):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.tnet1 = Tnet(3, 3)
        self.tnet2 = Tnet(64, 64)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 1024, kernel_size=1)
        self.conv3 = nn.Conv1d(1088, 128, kernel_size=1) # 1024 (global) + 64 (local)
        self.conv4 = nn.Conv1d(128, self.output_dims, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batch_size, _, num_pts = x.shape

        trans1 = self.tnet1(x)                                    # (B, 3, 3)
        skipped_x1 = x.transpose(2, 1)                            # (B, N, 3) 
        x = torch.bmm(skipped_x1, trans1)                         # (B, N, 3) x (B, 3, 3) = (B, N, 3)
        x = x.transpose(2, 1)                                     # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))                       # (B, 64, N)
                  
        local_feat = x                                            # (B, 64, N)  
        trans2 = self.tnet2(x)                                    # (B, 64, 64)
        skipped_x2 = x.transpose(2, 1)                            # (B, N, 64)
                      
        x = torch.bmm(skipped_x2, trans2)                         # (B, N, 64) x (B, 64, 64) = (B, N, 64) 
        x = x.transpose(2, 1)                                     # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))                       # (B, 1064, N)

        # Max pool:
        global_feat = torch.max(x, dim=2, keepdim=True) [0]       # (B, 1024, 1)
        global_feat = global_feat.repeat(1, 1, num_pts)           # (B, 1024, N)
        x = torch.cat([global_feat, local_feat], dim=1)           # (B, 1088, N)
        x = F.relu(self.bn3(self.conv3(x)))                       # (B, 128, N)
        x = self.conv4(x)                                         # (B, output_dims, N)
        return x



class PointNet(BaseModel):
    def __init__(self, input_channels=1, output_channels=1, **kwargs):
        super().__init__(name="UNet", **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels

    def create_model(self):
        self.model = PointNetTorch(
            input_dims = self.input_channels,
            output_dims= self.output_channels
        )
        return self.model