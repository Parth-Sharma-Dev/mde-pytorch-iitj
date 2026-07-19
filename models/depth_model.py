import torch
import torch.nn as nn
from .encoder import ResNetEncoder
from .decoder import UpConvBlock

class MonocularDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder(freeze_weights=True)
        
        self.upconv1 = UpConvBlock(in_channels=2048, skip_channels=1024, out_channels=512)
        self.upconv2 = UpConvBlock(in_channels=512, skip_channels=512, out_channels=256)
        self.upconv3 = UpConvBlock(in_channels=256, skip_channels=256, out_channels=128)
        self.upconv4 = UpConvBlock(in_channels=128, skip_channels=64, out_channels=64)
        
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1) 
        )

    def forward(self, x):
        encoded, skips = self.encoder(x)
        
        d = self.upconv1(encoded, skips[0])
        d = self.upconv2(d, skips[1])
        d = self.upconv3(d, skips[2])
        d = self.upconv4(d, skips[3])
        
        depth_map = self.final_upsample(d)
        
        return torch.relu(depth_map)