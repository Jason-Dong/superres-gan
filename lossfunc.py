from torch import nn
import torch
from torch.autograd import Variable

class PerceptualLoss(nn.Module):

    """
    Loss network is like VGG relu2_2
    """

    def __init__(self, loss_network):
        super(PerceptualLoss, self).__init__()
        self.loss_network = loss_network

    def forward(self, input, target):
        input_feature = self.loss_network(input)
        target_feature = self.loss_network(target)
        return torch.mean((input_feature - target_feature) ** 2)
