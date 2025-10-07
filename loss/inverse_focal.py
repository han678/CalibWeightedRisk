import torch.nn as nn
import torch.nn.functional as F
import torch

class InverseFocalLoss(nn.Module):
    def __init__(self, gamma=0, reduction='mean', is_prob=False):
        super(InverseFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.is_prob = is_prob
        self.eps = 1e-9

    def forward(self, input, target):
        # If input has more than 2 dimensions, flatten it (e.g., for images: N, C, H, W)
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)

        if self.is_prob:
            input = input.clamp(min=self.eps, max=1 - self.eps)
            logpt = torch.log(input)  # Use log of softmaxed input
        else:
            logpt = F.log_softmax(input, dim=1)  # Apply log_softmax to logits

        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.clamp(logpt.exp(), min=self.eps, max=1 - self.eps)
        loss = -1 * (1 + pt) ** self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
