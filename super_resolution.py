import torch
from torch import nn
from .modules import ConvBlock


class BasicSuperResolution(nn.Module):
    def __init__(self):
        super(BasicSuperResolution, self).__init__()

        self.layer1 = self.make_layer(1024, 512, 3)
        self.layer2 = self.make_layer(512, 128, 5)
        self.layer3 = self.make_layer(128, 64, 5)
        self.layer4 = self.make_layer(64, 3, 3)

    def make_layer(self, x, output, num_blocks):
        layers = []

        layers.append(ConvBlock(x, output, (3, 3), (1, 1)))

        for _ in range(1, num_blocks):
            layers.append(ConvBlock(output, output, (3, 3), (1, 1)))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.layer1(x)
        output = nn.functional.interpolate(output, scale_factor=(2, 2), mode='bilinear')

        output = self.layer2(output)
        output = nn.functional.interpolate(output, scale_factor=(2, 2), mode='bilinear')

        output = self.layer3(output)
        output = nn.functional.interpolate(output, scale_factor=(2, 2), mode='bilinear')

        output = self.layer4(output)
        output = nn.functional.interpolate(output, scale_factor=(2, 2), mode='bilinear')

        return output
