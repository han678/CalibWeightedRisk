import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

def kl_divergence(p, q):
    """Calculate the Kullback-Leibler (KL) divergence between two probability distributions."""
    return torch.sum(p * torch.log(p / q))

class ECELoss(torch.nn.Module):
    """
    Compute ECE (Expected Calibration Error)
    """

    def __init__(self, p=1, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.p = p
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logit: Tensor, target: Tensor) -> Tensor:
        scores = F.softmax(logit, dim=1)
        confidences, predictions = torch.max(scores, 1)
        accuracies = predictions.eq(target)
        ece = torch.zeros(1, device=logit.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += prop_in_bin * (torch.abs(avg_confidence_in_bin - accuracy_in_bin)) ** self.p
        return ece

class ECELoss_KL(nn.Module):
    """
    Compute ECE (Expected Calibration Error)
    """
    def __init__(self, n_bins=15):
        super(ECELoss_KL, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)

        num_classes = softmaxes.shape[1]
        per_class_ce = None

        for i in range(num_classes):
            classwise_y = (labels == i).long()
            gx_i = softmaxes[:, i].unsqueeze(-1)
            classwise_gx = torch.cat((1 - gx_i, gx_i), dim=1)
            classwise_y_onehot = nn.functional.one_hot(classwise_y, num_classes=classwise_gx.shape[1])

            class_ce = torch.zeros(1, device=softmaxes.device)
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = (gx_i.gt(bin_lower.item()) * gx_i.le(bin_upper.item())).squeeze()
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    cond_expect_in_bin = classwise_y_onehot[in_bin].float().mean(dim=0)
                    points_in_bin = classwise_gx[in_bin]
                    cond_expect_in_bin = torch.clamp(cond_expect_in_bin, min=1e-45)

                    for point in points_in_bin:
                        point = torch.clamp(point, min=1e-45)
                        class_ce += kl_divergence(cond_expect_in_bin, point)

            class_ce = (class_ce / len(gx_i))

            if i == 0:
                per_class_ce = class_ce
            else:
                per_class_ce = torch.cat((per_class_ce, class_ce), dim=0)

        return torch.mean(per_class_ce)