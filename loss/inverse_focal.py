import torch.nn as nn
import torch.nn.functional as F
import torch

class InverseFocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False, input_is_softmax=False):
        super(InverseFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.input_is_softmax = input_is_softmax
        self.eps = 1e-9

    def forward(self, input, target):
        # If input has more than 2 dimensions, flatten it (e.g., for images: N, C, H, W)
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)

        if self.input_is_softmax:
            input = input.clamp(min=self.eps, max=1 - self.eps)
            logpt = torch.log(input)  # Use log of softmaxed input
        else:
            logpt = F.log_softmax(input, dim=1)  # Apply log_softmax to logits

        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.clamp(logpt.exp(), min=self.eps, max=1 - self.eps)
        loss = -1 * (1 + pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
