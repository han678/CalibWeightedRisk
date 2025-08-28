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
    def __init__(self, gamma=None, size_average=False, is_prob=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.is_prob = is_prob
        self.eps = 1e-9

    def forward(self, input, target):
        # validate_input(input, target)
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        if self.is_prob:
            logpt = torch.log(input) 
        else:
            logpt = F.log_softmax(input, dim=1)  # Apply log_softmax to logits
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        # Set gamma based on pt
        if self.gamma is None:
            gamma = torch.where(pt < 0.2, torch.tensor(5.0, device=pt.device), torch.tensor(3.0, device=pt.device))
        else:
            gamma = self.gamma

        loss = -1 * (1 - pt) ** gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def validate_input(logits, labels):
    # logits: [batch, num_classes]
    # labels: [batch]
    num_classes = logits.shape[1]
    assert labels.dim() == 1, f"labels should be 1D, got {labels.shape}"
    assert labels.min() >= 0 and labels.max() < num_classes, f"labels out of range: {labels.min()} ~ {labels.max()}, num_classes={num_classes}"
    assert logits.shape[0] == labels.shape[0], f"batch size mismatch: logits {logits.shape}, labels {labels.shape}"

