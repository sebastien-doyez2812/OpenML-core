import torch
import torch.nn as nn
import torch.nn.functional as F

class Tnet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Tnet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Shared MLP layer = con1d
        self.conv1 = nn.Conv1d(self.input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.batch_norm5 = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.output_dim * self.output_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]

        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))

        x = torch.max(x, dim=2)[0]
        x = x.view(-1, 256)

        x = F.relu(self.batch_norm4(self.fc1(x)))
        x = F.relu(self.batch_norm5(self.fc2(x)))
        x = self.fc3(x)

        identity_matrix = torch.eye(self.output_dim, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1) # (B, output_dim, output_dim)
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x