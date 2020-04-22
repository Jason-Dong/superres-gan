import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
     def __init__(self, upscale_factor=2):
        super(PixelShuffle, self).__init__()
        self.upsample = nn.PixelShuffle(upscale_factor)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        return F.relu(x)
