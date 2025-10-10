'''
Implementation of Focal Loss.
Reference:
[1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
     arXiv preprint arXiv:1708.02002, 2017.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=None, reduction='mean', is_prob=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.is_prob = is_prob
        self.eps = 1e-9

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        if self.is_prob:
            input = torch.clamp(input, min=self.eps, max=1 - self.eps)  # 防止 log(0)
            logpt = torch.log(input)
        else:
            logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        if self.gamma is None:
            gamma = torch.where(pt < 0.2, torch.tensor(5.0, device=pt.device), torch.tensor(3.0, device=pt.device))
        else:
            gamma = self.gamma
        loss = -1 * (1 - pt) ** gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss