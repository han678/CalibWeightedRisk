import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss

from loss.aurc import get_score_function
from .hist_rank import DifferentiableHistogramCDF

def neg_entropy_normalized(neg_entropy, k):
    """compute normalized negative entropy"""
    max_neg_entropy = 0.0
    min_neg_entropy = torch.log(torch.tensor(k, dtype=neg_entropy.dtype, device=neg_entropy.device))  # 最大熵
    normalized_neg_entropy = (neg_entropy - min_neg_entropy) / (max_neg_entropy - min_neg_entropy)
    return torch.clamp(normalized_neg_entropy, min=0.0, max=1.0)

def identity(x, k):
    return x

class SelectiveAULoss(_Loss):
    """
    Selective AU Loss: Computes weighted loss for all samples, 
    where samples below the quantile (e.g. lowest 15% confidence) 
    are weighted by their rank, and samples above the quantile 
    are weighted by the quantile rank. 
    This encourages the model to focus more on low-confidence (uncertain) samples.
    """
    def __init__(self, batch_size=128, reduction='sum',
                 hist_edges=None, hist_sigma=0.05, hist_device=None,
                 score_function="MSP", is_prob=False,
                 conf_quantile=None):
        super().__init__(reduction=reduction)
        self.batch_size = batch_size
        self.is_prob = is_prob
        self.score_func = get_score_function(score_function, self.is_prob)
        self.eps = 1e-9
        self.conf_quantile = conf_quantile
        if hist_edges is None:
            hist_edges = torch.linspace(0, 1, 65)
        self.hist_sigma = hist_sigma
        self.hist_cdf = DifferentiableHistogramCDF(hist_edges, sigma=hist_sigma, device=hist_device)
        if self.is_prob:
            self.criterion = torch.nn.NLLLoss(reduction='none')
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        if score_function == "NegEntropy":
            self.transform = neg_entropy_normalized
        else:
            self.transform = identity

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.is_prob:
            scores = torch.clamp(input, min=self.eps, max=1 - self.eps).to(input.device)
            log_scores = torch.log(scores)
            ce_loss = self.criterion(log_scores, target)
            confidence = self.score_func(scores)
        else:
            softmaxed = F.softmax(input, dim=1)
            scores = torch.clamp(softmaxed, min=self.eps, max=1 - self.eps).to(input.device)
            ce_loss = self.criterion(input, target)
            confidence = self.transform(self.score_func(scores),k=softmaxed.shape[1]).to(input.device)
        if torch.isnan(confidence).any():
            print("Warning: NaN in confidence scores")
            confidence = torch.where(torch.isnan(confidence), torch.zeros_like(confidence), confidence)
        self.hist_cdf.add(confidence)
        Gx = self.hist_cdf.cdf(confidence)
        # compute tau and G(tau)
        if self.conf_quantile is not None:
            tau = torch.quantile(confidence, self.conf_quantile)
            Gtau = self.hist_cdf.cdf(tau)
        else:
            # Default behavior: use confidence quantile
            tau = torch.quantile(confidence, 0.5)
            Gtau = self.hist_cdf.cdf(tau)
        Gx = torch.clamp(Gx, min= 1 / (self.batch_size + 1),  max=self.batch_size / (self.batch_size + 1))
        Gtau = torch.clamp(Gtau, min= 1 / (self.batch_size + 1), max=self.batch_size / (self.batch_size + 1))
        const_alpha = -torch.log(1 - Gtau).to(input.device)
        alpha = torch.where(
            (confidence >= tau).to(input.device),
            const_alpha.to(input.device),
            -torch.log(1 - Gx).to(input.device)
        )
        # Final selective AU loss
        losses = ce_loss * alpha
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses

    def plot_cdf(self, save_path=None):
        """
        Visualize the currently accumulated histogram CDF curve.
        If save_path is not None, save the image; otherwise, display it directly.
        """
        import matplotlib.pyplot as plt

        centers = self.hist_cdf.centers.detach().cpu().numpy()
        cdf_vals = self.hist_cdf.cdf(self.hist_cdf.centers).detach().cpu().numpy()

        plt.figure(figsize=(6, 4))
        plt.plot(centers, cdf_vals, label='CDF estimator', color='blue')
        plt.xlabel('Confidence')
        plt.ylabel('CDF')
        plt.xlim(0, 1)
        # plt.title('Histogram-based CDF')
        plt.grid(True)
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()