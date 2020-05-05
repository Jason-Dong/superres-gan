import torch
import torch.nn as nn

import torchvision.models as models


class DenseNetBackbone(nn.Module):
    def __init__(self):
        super(DenseNetBackbone, self).__init__()

        self.model = models.densenet121(pretrained=True)

    def forward(self, x):
        output = self.model.features.conv0(x)

        output = self.model.features.norm0(output)
        output = self.model.features.relu0(output)
        output = self.model.features.pool0(output)

        output = self.model.features.denseblock1(output)

        output = self.model.features.transition1(output)

        output = self.model.features.denseblock2(output)

        output = self.model.features.transition2(output)

        output = self.model.features.denseblock3(output)

        return output
