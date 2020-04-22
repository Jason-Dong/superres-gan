import torch
import torch.nn as nn

#Using PReLU instead of leaky relu since it learns the parameter
# 'a' instead of using 0.01 like Leaky relu

class ConvPrelu(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvPrelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.prelu(self.conv(x))
        return x
