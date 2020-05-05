"""Contains some basic models for training"""
import torch
import torch.nn as nn

from torchvision.models import resnet101
from .modules import ConvBlock


class SegmentationModelResnet(nn.Module):
    """Basic model for testing
       Compatible with resnet_backbone"""
    def __init__(self):
        super(SegmentationModelResnet, self).__init__()
        self.classes = 19

        self.conv1 = ConvBlock(2048, 256, 3, padding=(1, 1))
        self.conv1_2 = ConvBlock(256, 256, 3, padding=(1, 1))
        self.conv2 = ConvBlock(256, 128, 3, padding=(1, 1))
        self.conv2_2 = ConvBlock(128, 128, 3, padding=(1, 1))
        self.conv3 = ConvBlock(128, 64, 3, padding=(1, 1))
        self.conv3_2 = ConvBlock(64, 64, 3, padding=(1, 1))
        self.conv4 = ConvBlock(64, self.classes, 3, padding=(1, 1))

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv1_2(output)
        output = nn.functional.interpolate(output, scale_factor=(4, 4), mode='bilinear')

        output = self.conv2(output)
        output = self.conv2_2(output)
        output = nn.functional.interpolate(output, scale_factor=(2, 2), mode='bilinear')

        output = self.conv3(output)
        output = self.conv3_2(output)
        output = nn.functional.interpolate(output, scale_factor=(2, 2), mode='bilinear')

        output = self.conv4(output)
        output = nn.functional.interpolate(output, scale_factor=(2, 2), mode='bilinear')

        return output


class SegmentationModelDenseNet(nn.Module):
    """Segmentation model compatible with DenseNet Backbone"""
    def __init__(self):
        super(SegmentationModelDenseNet, self).__init__()
        self.classes = 19

        self.conv1 = ConvBlock(1024, 512, 3, padding=(1, 1))
        self.conv1_1 = ConvBlock(512, 256, 3, padding=(1, 1))
        self.conv1_2 = ConvBlock(256, 256, 3, padding=(1, 1))
        self.conv2 = ConvBlock(256, 128, 3, padding=(1, 1))
        self.conv2_2 = ConvBlock(128, 128, 3, padding=(1, 1))
        self.conv3 = ConvBlock(128, 64, 3, padding=(1, 1))
        self.conv3_2 = ConvBlock(64, 64, 3, padding=(1, 1))
        self.conv4 = ConvBlock(64, self.classes, 3, padding=(1, 1))
        self.conv4_1 = ConvBlock(self.classes, self.classes, 3, padding=(1, 1))

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv1_1(output)
        output = self.conv1_2(output)
        output = nn.functional.interpolate(output, scale_factor=(2, 2), mode='bilinear')

        output = self.conv2(output)
        output = self.conv2_2(output)
        output = nn.functional.interpolate(output, scale_factor=(2, 2), mode='bilinear')

        output = self.conv3(output)
        output = self.conv3_2(output)
        output = nn.functional.interpolate(output, scale_factor=(2, 2), mode='bilinear')

        output = self.conv4(output)
        output = nn.functional.interpolate(output, scale_factor=(2, 2), mode='bilinear')
        output = self.conv4_1(output)

        return output
