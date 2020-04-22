import torch
import torch.nn as nn

class ConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvBatchRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.batchnorm(x))
        return x
