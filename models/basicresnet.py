import torch
import torch.nn as nn
import torch.nn.functional as F

#A basic resnet block we'll use.
class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, padding=0):
        super(ResBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_input = x
        x = self.conv3x3(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = self.batchnorm(x)
        return x + x_input
