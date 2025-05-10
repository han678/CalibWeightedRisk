import torch.nn.functional as F
import torch.nn as nn
import torch


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='sum'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, input, target):
        num_classes = input.size(-1) # check
        target_bi = torch.zeros(input.size(0), num_classes).scatter_(1, target.view(-1, 1).long(), 1)
        target_bi_smooth = (1.0 - self.epsilon) * target_bi + self.epsilon / num_classes
        loss = -torch.sum(torch.nn.functional.log_softmax(input, dim=1) * target_bi_smooth, dim=1)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return loss
