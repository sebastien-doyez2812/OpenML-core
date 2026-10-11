import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .base_model import BaseModel

class MobileNetBlockv2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)
        
        hidden_dim = in_channels * expand_ratio
        self.layers = nn.ModuleList()

        if expand_ratio != 1:
            self.layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            self.layers.append(nn.BatchNorm2d(hidden_dim))
            self.layers.append(nn.ReLU6(inplace=True)) # ReLU6 est souvent préféré pour MobileNet
        else:
            hidden_dim = in_channels

        self.layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=self.stride, padding=1, groups=hidden_dim, bias=False))
        self.layers.append(nn.BatchNorm2d(hidden_dim))
        self.layers.append(nn.ReLU6(inplace=True))

        self.layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        self.layers.append(nn.BatchNorm2d(out_channels))

    def forward(self, X):
        inputX = X if self.use_residual else None
        
        for layer in self.layers:
            X = layer(X)

        if inputX is not None:
            return inputX + X
        return X
    
class STEM(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__(**kwargs)
        
        self.conv2D = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchNorm = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)
        
        self.mobile_block1  = MobileNetBlockv2(in_channels=32, out_channels=16, stride=1, expand_ratio=1)
        self.mobile_block2a = MobileNetBlockv2(in_channels=16, out_channels=24, stride=2, expand_ratio=6)
        self.mobile_block2b = MobileNetBlockv2(in_channels=24, out_channels=24, stride=1, expand_ratio=6)
        self.mobile_block3  = MobileNetBlockv2(in_channels=24, out_channels=24, stride=1, expand_ratio=6)

    def forward(self, X):
        X = self.relu(self.batchNorm(self.conv2D(X)))
        X = self.mobile_block1(X)
        X = self.mobile_block2a(X)
        X = self.mobile_block2b(X)
        X = self.mobile_block3(X)
        return X

class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=2, l=2, dim=64, **kwargs):
        super().__init__(**kwargs)
        self.patchsize = patch_size
        self.dim = dim
        self.out_channels = out_channels
        self.L = l
        
        self.conv3x3_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False)
        self.conv1x1_1 = nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=1, padding=0, bias=False)
        
        self.conv3x3_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.conv1x1_2 = nn.Conv2d(in_channels=dim, out_channels=in_channels, kernel_size=1, padding=0, bias=False)
        
        self.transformers = nn.ModuleList([ViTransformer(dim=self.patchsize * self.patchsize * dim) for _ in range(self.L)])
        
        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

    def forward(self, X):
        skip_connection = X

        X = self.conv3x3_1(X)
        X = self.conv1x1_1(X)

        B, C, H, W = X.shape
        pH, pW = H // self.patchsize, W // self.patchsize
        
        X = self.unfold(X)
        X = X.reshape(B, C * self.patchsize * self.patchsize, pH * pW)
        X = X.permute(0, 2, 1)

        for transformer in self.transformers:
            X = transformer(X)

        X = X.permute(0, 2, 1)
        X = X.reshape(B, C, self.patchsize, self.patchsize, pH, pW)
        X = X.permute(0, 1, 4, 2, 5, 3)
        X = X.reshape(B, C, pH * self.patchsize, pW * self.patchsize)

        X = self.conv1x1_2(X)
        X = X + skip_connection
        X = self.conv3x3_2(X)
        
        return X
    
class ViTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_ratio=4, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.dropout_rate = dropout

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn_1 = nn.Linear(in_features=dim, out_features=dim * ffn_ratio)
        self.gelu = nn.GELU()
        self.ffn_2 = nn.Linear(in_features=dim * ffn_ratio, out_features=dim)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, X):
        input_tensor = X
        X = self.norm1(X)
        X, _ = self.attn(X, X, X)
        X = self.drop1(X)
        X = X + input_tensor

        res = X
        X = self.norm2(X)
        X = self.ffn_1(X)
        X = self.gelu(X)
        X = self.drop2(X)
        X = self.ffn_2(X)
        X = self.drop2(X)
        return X + res
    
class HybridEncoderBlock(nn.Module):
    def __init__(self, input_channels, out_channels, stride=2, **kwargs):
        super().__init__(**kwargs)
        self.Conv1x1 = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.Conv3x3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.mobileViT_block = MobileViTBlock(in_channels=out_channels, out_channels=out_channels)

    def forward(self, X):
        X = self.Conv1x1(X)
        X = self.Conv3x3(X)
        X = self.mobileViT_block(X)
        return X

class HybridDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=2, 
            stride=2
        )
        
        concat_channels = out_channels + skip_channels
        
        self.conv1x1 = nn.Conv2d(
            in_channels=concat_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        self.silu1 = nn.SiLU()
        
        self.conv_depth_wise = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            groups=out_channels, 
            bias=False
        )
        self.silu2 = nn.SiLU()
        
        self.conv1x1_2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        self.silu3 = nn.SiLU()
        
        self.mobileViTBlock = MobileViTBlock(
            in_channels=out_channels, 
            out_channels=out_channels
        )

    def forward(self, decoder_in, skip_in):
        x = self.transpose_conv(decoder_in)
        
        if x.shape[2:] != skip_in.shape[2:]:
            x = F.interpolate(x, size=skip_in.shape[2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([x, skip_in], dim=1)
        x = self.silu1(self.conv1x1(x))
        x = self.silu2(self.conv_depth_wise(x))
        x = self.silu3(self.conv1x1_2(x))
        return self.mobileViTBlock(x)

class MobileUnetRTorch(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.STEM = STEM(in_channels=input_channels) #should be 3, because RGB = 3 channels
        # HEB = Hybrid Encoder Block
        self.HEB_1 = HybridEncoderBlock(input_channels=24, out_channels=64,  stride=2)          # H/8
        self.HEB_2 = HybridEncoderBlock(input_channels=64, out_channels=128,  stride=2)          # H/16
        self.HEB_3 = HybridEncoderBlock(input_channels=128, out_channels=256,  stride=2)          # H/32
        self.HEB_4 = HybridEncoderBlock(input_channels=256, out_channels=512,  stride=2)          # H/64

        # HDB = Hybrid Decoder Block
        self.HDB_1 = HybridDecoderBlock(in_channels= 512, skip_channels=256, out_channels=256) #H/32
        self.HDB_2 = HybridDecoderBlock(in_channels= 256, skip_channels=128, out_channels=128) #H/16
        self.HDB_3 = HybridDecoderBlock(in_channels= 128, skip_channels=64, out_channels=64)   #H/8

        self.final_conv = nn.ConvTranspose2d(in_channels=64, out_channels=output_channels, kernel_size=4, stride=4, padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self, X):
        input_size = X.shape[2:]
        skip_connection = []
        
        X = self.STEM(X) # Embedding
        skip_connection.append(X)
        X = self.HEB_1(X)
        skip_connection.append(X)

        X = self.HEB_2(X)
        skip_connection.append(X)

        X = self.HEB_3(X)
        skip_connection.append(X)

        X = self.HEB_4(X)

        X = self.HDB_1(X, skip_connection.pop())
        X = self.HDB_2(X, skip_connection.pop())
        X = self.HDB_3(X, skip_connection.pop())

        X = self.final_conv(X)
        
        if X.shape[2:] != input_size:
            X = F.interpolate(X, size=input_size, mode='bilinear', align_corners=False)
        return X  
        # return self.sigmoid(X)           

class MobileUnetRTorch_v2(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        
        self.encoder = timm.create_model(
            "mobilevit_s.cvnets_in1k", 
            pretrained=True, 
            features_only=True, 
            out_indices=(1, 2, 3, 4)
        )
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        # features[3] (bottleneck) = 640 channels
        # features[2] (skip3)      = 128 channels
        # features[1] (skip2)      = 96 channels
        # features[0] (skip1)      = 64 channels
        
        self.HDB_1 = HybridDecoderBlock(in_channels=640, skip_channels=128, out_channels=256)
        self.HDB_2 = HybridDecoderBlock(in_channels=256, skip_channels=96,  out_channels=128)
        self.HDB_3 = HybridDecoderBlock(in_channels=128, skip_channels=64,  out_channels=64)

        self.final_conv = nn.ConvTranspose2d(in_channels=64, out_channels=output_channels, kernel_size=4, stride=4, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        input_size = X.shape[2:]
        
        features = self.encoder(X)
        skip1, skip2, skip3, bottleneck = features[0], features[1], features[2], features[3]

        X = self.HDB_1(bottleneck, skip3)
        X = self.HDB_2(X, skip2)
        X = self.HDB_3(X, skip1)

        X = self.final_conv(X)
        
        if X.shape[2:] != input_size:
            X = F.interpolate(X, size=input_size, mode='bilinear', align_corners=False)

        return X  
        # return self.sigmoid(X)

class MobileUnetR(BaseModel):
    def __init__(self, input_channels=1, output_channels=1, **kwargs):
        super().__init__(name="MobileUnetR", **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels

    def create_model(self):
        self.model = MobileUnetRTorch(
            input_channels=self.input_channels,
            output_channels=self.output_channels
        )
        return self.model

class MobileUnetR_v2(BaseModel):
    def __init__(self, input_channels=1, output_channels=1, **kwargs):
        super().__init__(name="MobileUnetR_apple_version", **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels

    def create_model(self):
        self.model = MobileUnetRTorch_v2(
            input_channels=self.input_channels,
            output_channels=self.output_channels
        )
        return self.model