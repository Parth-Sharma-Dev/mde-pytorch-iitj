import torch
import torch.nn as nn

class UpConvBlock(nn.Module):
    def _init_(self, in_channels, skip_channels, out_channels):
        super()._init_()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        total_channels = in_channels + skip_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)