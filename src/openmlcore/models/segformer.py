import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .lib.convolutions import DoubleConv
import math

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads = 8, reduction_ratio = 1):
        super().__init__()
        self.dim = dim # = C in the article
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.reduction_ratio = reduction_ratio

        self.W_q = nn.Linear(dim, dim)
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

        self.W_k  = nn.Linear(dim, dim)
        self.W_v  = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, X, H, W):
        B, N, C = X.shape # N = h * W
        # (B, N, C) -> (B, N, heads, head_dim) -> (B, heads, N, head_dim)
        q = self.W_q(X).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.sr is not None:
            x_spatial = X.permute(0, 2, 1). reshape(B, C, H, W)
            x_spatial = self.sr(x_spatial)
            x_spatial = x_spatial.flatten(2) # B, C, H*W
            x_spatial = x_spatial.permute(0, 2, 1)
            x_spatial = self.norm(x_spatial)

            k = self.W_k(x_spatial).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.W_v(x_spatial).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            # Classical Attention
            k = self.W_k(X).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.W_v(X).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_value = (q @ k.transpose(-2, -1)) * self.scale
        attn_value = attn_value.softmax(dim=-1)

        # (B, heads, N, N_reduced) @ v: (B, heads, N_reduced, head_dim) -> (B, heads, N, head_dim)
        X_out = attn_value @ v
        X_out = X_out.permute(0, 2, 1, 3). reshape(B, N, C)
        return self.proj(X_out)

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = nn.LayerNorm(out_channels)
    def forward(self, X):
        # B, C, H, W
        X = self.conv(X)
        B, C, H_new, W_new = X.shape

        X = X.flatten(2). transpose(2, 1) # B, N, C
        X = self.norm(X)
        return X, H_new, W_new


class TransformerSegFormer(nn.Module):
    def __init__(self, N, C_in, C_out, num_heads, reduction_ratio = 1, intermediate_channel = 256, H = 256, W = 256, patch_kernel = 3, stride = 2, padding = 1):
        super().__init__()

        self.efficient_self_attns = nn.ModuleList()
        self.mix_FFNs = nn.ModuleList()
        self.N = N

        for i in range(N):
            self.efficient_self_attns.append(EfficientSelfAttention(dim = C_out, num_heads= num_heads, reduction_ratio= reduction_ratio))
            target_H = H // stride
            target_W = W // stride
            self.mix_FFNs.append(MixFFN(C_out, intermediate_channel, H=target_H, W=target_W))

        self.overlap_patch_merging = OverlapPatchMerging(C_in, C_out,patch_kernel, stride = stride, padding = padding)

    def forward(self, X):
        X, H_new, W_new = self.overlap_patch_merging(X)

        for idx in range(self.N):
            X = self.mix_FFNs[idx](X + self.efficient_self_attns[idx](X, H_new, W_new))

        return X, H_new, W_new
        
class MixFFN(nn.Module):
    def __init__(self, in_channels, intermediate_channels, H=256, W=256):
        super().__init__()
        self.H = H
        self.W = W

        self.MLP1 = nn.Linear(in_features=in_channels, out_features=intermediate_channels)
        self.gelu = nn.GELU()
        self.conv33 = nn.Conv2d(in_channels=intermediate_channels, out_channels= intermediate_channels, kernel_size=3, padding=1)
        self.MLP2 = nn.Linear(in_features=intermediate_channels, out_features=in_channels)

    def forward(self, X):
        skipped_X = X
        B, N, C = X.shape
        # X = B, N, C, due to the efficient self Attention
        # Linear works with B, N, C, we need to permute et flatten
        X = self.MLP1(X) # OK
        X = self.gelu(X)

        # Conv2D needs B, C, H, W
        X = X.view(B, self.H, self.W, -1) # B, H, W, C
        X = X.permute(0, 3, 1, 2)
        X = self.conv33(X)

        # Linear / MLP needs B, N, C
        X = X.flatten(2) # Flatten applied for the dim >=2 (H & W) 
        # X = B, C, N 
        X = self.MLP2(X.permute(0, 2, 1))
        return X + skipped_X

class MLPLayers(nn.Module):
    def __init__(self, i, input_channels, embbeding_channels, H = 256, W = 256):
        super().__init__()
        # Keep the same dimension but project on the number of channels:
        self.linear_project = nn.Conv2d(in_channels= input_channels, out_channels= embbeding_channels, kernel_size=1)
        
        
        self.current_H = H // (2**(i + 2))
        self.current_W = W // (2**(i + 2))
        self.target_H = H // 4
        self.target_W = W // 4

        self.linear_spatial = nn.Linear((self.current_H * self.current_W), self.target_H * self.target_W)

    def forward(self, X):
        # B, Ci, H/2**(i+1), W/2**(i+1)
        B, _, _, _ = X.shape

        # Projection on C:
        #  B, C, H/2**(i+1), W/2**(i+1)
        # C is embbeding channels:
        X = self.linear_project(X) 

        #  B, C, H/2**(i+1)* W/2**(i+1)
        X = X.flatten(2)

        #  B, C, H/4* W/4
        X = self.linear_spatial(X)

        #  B, C, H/4, W/4
        X = X.view(B, -1, self.target_H, self.target_W)
        return X

    
class SegFormerTorch(nn.Module):
    def __init__(self, depth = 4, input_channels=1, output_channels=1, initial_filters=64, H=256, W= 256):
        super().__init__()
        self.depth = depth
        self.channels = [initial_filters*2**i for i in range(self.depth)]
        self.num_heads = [1, 2, 4, 8]
        reduction_ratios = [8, 4, 2, 1]
        
        self.transformers = nn.ModuleList()
        current_in_channels = input_channels
        
        current_H, current_W = H, W

        for i in range(self.depth):
            patch_kernel = 7 if i == 0 else 3
            patch_stride = 4 if i == 0 else 2
            patch_padding = 3 if i == 0 else 1


            self.transformers.append(TransformerSegFormer(
                                        N=2, # Nombre de blocs Attention+MixFFN par niveau
                                        C_in=current_in_channels,
                                        C_out=self.channels[i],
                                        num_heads=self.num_heads[i],
                                        reduction_ratio=reduction_ratios[i],
                                        intermediate_channel=self.channels[i] * 4,
                                        H=current_H,
                                        W=current_W,
                                        patch_kernel=patch_kernel,
                                        stride=patch_stride,
                                        padding=patch_padding
                                        )
                                    )
            current_in_channels = self.channels[i]
            current_H //= patch_stride
            current_W //= patch_stride
        self.embedding_dim = 256

        self.mlp_layers = nn.ModuleList()
        for i in range(self.depth):
            self.mlp_layers.append(
                MLPLayers(i=i, input_channels=self.channels[i], embbeding_channels=self.embedding_dim, H=H, W=W)
            )
        self.linear_fuse = nn.Conv2d(self.embedding_dim * self.depth, self.embedding_dim, kernel_size=1)
        self.linear_pred = nn.Conv2d(self.embedding_dim, output_channels, kernel_size=1)

    def forward(self, X):
        orig_H, orig_W = X.shape[-2], X.shape[-1]
        features = []
        for idx in range(self.depth):
            X, H_new, W_new = self.transformers[idx](X)
            X_spatial = X.transpose(1, 2).view(X.shape[0], -1, H_new, W_new).contiguous()
            features.append(X_spatial)
            X = X_spatial

        decoder_output = []
        for idx in range(self.depth):
            decoder_output.append(self.mlp_layers[idx](features[idx]))

        out = torch.cat(decoder_output, dim = 1)
        out = self.linear_fuse(out)
        out = F.gelu(out)
        out = self.linear_pred(out)
        # To get the same dimension as the input:
        out = F.interpolate(out, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
        
        return out

class SegFormer(BaseModel):
    def __init__(self, input_channels=1, output_channels=1, H = 256, W = 256,depth=4, initial_filters=64, **kwargs):
        super().__init__(name="SegFormer", **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depth = depth
        self.initial_filters = initial_filters
        self.H = H
        self.W = W

    def create_model(self):
        self.model = SegFormerTorch(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            depth=self.depth,
            initial_filters=self.initial_filters,
            H = self.H,
            W = self.W
        )
        return self.model