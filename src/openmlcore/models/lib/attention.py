import torch
from torch import nn

class AttentionGate(nn.Module):
    def __init__(self, in_channels_encoder, in_channels_decoder, interchannel): 
        super().__init__()
        self.conv_enc = nn.Conv2d(in_channels_encoder, interchannel, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_dec = nn.Conv2d(in_channels_decoder, interchannel, kernel_size=1, stride=1, padding=0, bias=True)
        self.final_conv = nn.Conv2d(interchannel, 1, kernel_size=1, stride=1, padding=0, bias=True )

        self.sigmoid = nn.Sigmoid()
        self.relu    = nn.ReLU(inplace=True)  

    def forward(self, encoder, decoder):
        enc = self.conv_enc(encoder)
        dec = self.conv_dec(decoder)
        sum = self.relu(enc + dec)
        x = self.sigmoid(self.final_conv(sum))
        
        return encoder * x

