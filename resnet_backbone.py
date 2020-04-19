"""Contains some basic models for training"""
import torch
import torch.nn as nn

from torchvision.models import resnet101
from .modules import ConvBlock


class ResnetModel(nn.Module):
    """Basic model for testing"""
    def __init__(self):
        super(ResnetModel, self).__init__()

        self.resnet = resnet101(pretrained=True, progress=True)

    def forward(self, x):
        print("backbone input: ", x.shape)
        output = self.resnet.conv1(x)
        output = self.resnet.bn1(output)
        output = self.resnet.relu(output)
        output = self.resnet.maxpool(output)

        print("first layer output: ", output.shape)

        output = self.resnet.layer1(output)
        print("second layer output: ", output.shape)

        output = self.resnet.layer2(output)
        print("third layer output: ", output.shape)

        output = self.resnet.layer3(output)

        print("fourth layer output: ", output.shape)
        output = self.resnet.layer4(output)
        print("fifth layer output: ", output.shape)

        return output
