import torch
import torch.nn as nn
import torch.nn.functional as F
from models.upsamplingblock import UpBlock
from models.basicresnet import ResBlock
from models.prelu import ConvPrelu
from models.batchrelu import ConvBatchRelu

class SuperRes(nn.Module):
    def __init__(self):
        super(SuperRes, self).__init__()
        self.batch_norm = nn.BatchNorm2d(3)
        self.conv_in = ConvBatchRelu(3, 64, kernel_size=9, padding=4)
        self.resblock = ResBlock(64, 64, padding=1)
        self.upblock = UpBlock(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=9, padding=4)
#compiles all the blocks.
    

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv_in(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.upblock(x)
        x = self.upblock(x)
        x = self.conv_out(x)
        return F.tanh(x)
