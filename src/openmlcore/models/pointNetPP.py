# PointNet++

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

def furthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance = torch.where(mask, dist, distance)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids

def query_ball_points(radius, nsample, xyz, new_xyz):
    device  = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    
    sqrdists = square_distance(new_xyz, xyz) 
    mask = sqrdists > radius ** 2
    group_idx = torch.where(mask, torch.tensor(N, device=device, dtype=torch.long), group_idx)

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx = torch.where(mask, group_first, group_idx)
    
    return group_idx

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    head = torch.sum(src ** 2, dim= -1, keepdim=True)
    tail = torch.sum(dst ** 2, dim=-1, keepdim=True).permute(0, 2, 1)

    return dist + head + tail

def index_points(points, idx):
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]  
    return new_points

class SetAbstraction(nn.Module):
    def __init__(self, npoints, radius, nsample, in_channel, mlp ):
        super().__init__()
        self.npoints = npoints
        self.radius = radius
        self.nsample = nsample
        self.inchannel = in_channel

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = 3 if in_channel == 3 else in_channel + 3

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        centroid_idx = furthest_point_sample(xyz, self.npoints)
        new_xyz = index_points(xyz, centroid_idx)

        group_idx = query_ball_points(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = index_points(points, group_idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        new_points = new_points.permute(0, 3, 1, 2)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=-1)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points

class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        # xyz2 has the less points/ low resolution
        # xyz1 has a higher resolution
        # points1: skip connection
        # points2: features of the previous layers, shape = B, S, C2

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1) 
        else:
            # Search for the 3 closers points in xyz2
            dists = square_distance(xyz1, xyz2) # (B, N, S)
            dists, idx = torch.topk(dists, 3, dim=-1, largest=False, sorted=True)

            weights = 1.0 / (dists + 1e-8)
            weight = weights / torch.sum(weights, dim=-1, keepdim=True) #(B, N, 3)

            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            if points1 is not None:
                new_points = torch.cat([points1, interpolated_points], dim=-1)
            else:
                new_points = interpolated_points

            new_points = new_points.permute(0, 2, 1)
            for conv, bn in zip(self.mlp_convs, self.mlp_bns):
                new_points = F.relu(bn(conv(new_points)))
            
            return new_points.permute(0, 2, 1)


class PointNetPPSemSegTorch(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # --- ENCODER (Set Abstraction ) ---
        self.sa1 = SetAbstraction(npoints=1024, radius=0.1, nsample=32, in_channel=3, mlp=[32, 32, 64])
        self.sa2 = SetAbstraction(npoints=256, radius=0.2, nsample=32, in_channel=64, mlp=[64, 64, 128])
        self.sa3 = SetAbstraction(npoints=64, radius=0.4, nsample=32, in_channel=128, mlp=[128, 128, 256])
        self.sa4 = SetAbstraction(npoints=16, radius=0.8, nsample=32, in_channel= 256, mlp=[256, 256, 512]) # Version sans features initiales directes (in_channel implicite)
        
        # --- DECODER (Feature Propagation) ---
        # Remonte progressivement de 16 -> 64 -> 256 -> 1024 -> N points d'origine
        self.fp4 = FeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp3 = FeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        self.fp2 = FeaturePropagation(in_channel=256 + 64, mlp=[256, 128])
        self.fp1 = FeaturePropagation(in_channel=128 + 3, mlp=[128, 128, 128])
        
        # --- TÊTE DE CLASSIFICATION PAR POINT ---
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        if xyz.shape[1] == 3:
            xyz = xyz.permute(0, 2, 1)

        l0_xyz = xyz
        l0_points = None
        
        # 1. Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        # 2. Décoder (Propagation of features + Skip Connections)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_xyz, l1_points) # Skip to the origin cloud l0_xyz
        
        # 3. Final classification
        net = l0_points.permute(0, 2, 1) # (B, C, N) for Conv1d
        net = F.relu(self.bn1(self.conv1(net)))
        net = self.drop(net)
        logits = self.conv2(net) # (B, num_classes, N)
        
        return logits #.permute(0, 2, 1) # return (B, N, num_classes)


class PointNetPP(BaseModel):
    def __init__(self, output_channels=1, **kwargs):
        super().__init__(name="PointNet++", **kwargs)
        self.output_channels = output_channels

    def create_model(self):
        self.model = PointNetPPSemSegTorch( num_classes= self.output_channels )
        return self.model