"""Calculates the Cross Entropy Loss"""
import torch
import torch.nn as nn
from .metrics import get_mIOU


class CrossEntropy(nn.Module):
    """Upscales image if necessary then computer cross entropy loss"""
    def __init__(self, ignore_index=255):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, input, label):
        loss = self.criterion(input, label.long())

        return loss
