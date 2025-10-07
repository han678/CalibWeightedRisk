import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import digamma
from torch import Tensor
from torch.nn.modules.loss import _Loss

def cross_entropy_loss(probs, targets):
    """Compute cross-entropy loss between prediction probabilities and targets."""
    eps = 1e-9  # Small value to avoid log(0)
    probs = np.clip(probs, eps, 1 - eps)
    ce_loss = -np.sum(targets * np.log(probs), axis=1)
    return ce_loss
    
def compute_harmonic_alphas(n):
    return [digamma(n + 1) - digamma(n - rank + 1) for rank in range(1, n + 1)]

def sele_alphas(n):
    return [2 * (rank / n) for rank in range(1, n + 1)]


# Utility functions for scoring

def entropy(x):
    eps = 1e-9
    x = torch.clamp(x, min=eps)
    return -torch.sum(x * torch.log(x), dim=1)

def top12_margin(x):
    values, _ = torch.topk(x, k=2, dim=-1)
    return values[:, 0] - values[:, 1] if x.ndim > 1 else values[0] - values[1]

def gini_score(x):
    return 1 - torch.norm(x, dim=1, p=2) ** 2

def get_score_function(name: str, is_prob: bool = False):
    eps = 1e-9
    as_probs = (lambda x: x) if is_prob else (lambda x: torch.clamp(F.softmax(x, dim=1), min=eps, max=1 - eps))
    mapping = {
        "MSP": lambda x: torch.max(as_probs(x), dim=1).values,
        "NegEntropy": lambda x: -entropy(as_probs(x)),
        "SoftmaxMargin": lambda x: top12_margin(as_probs(x)),
        "MaxLogit": lambda x: torch.max(as_probs(x), dim=1).values,
        "l2_norm": lambda x: torch.norm(as_probs(x), dim=1, p=2),
        "NegGiniScore": lambda x: -gini_score(as_probs(x)),
    }
    if name not in mapping:
        raise ValueError(f"Unknown score function: {name}")
    return mapping[name]



class BaseAURCLoss(_Loss):
    def __init__(self, score_function="MSP", batch_size=128, reduction='sum', alpha_fn=None, is_prob=False):
        super().__init__(reduction=reduction)
        self.batch_size = batch_size
        self.is_prob = is_prob
        self.score_func = get_score_function(score_function, self.is_prob)
        self.alphas = alpha_fn(batch_size) if alpha_fn else None
        self.eps = 1e-9
        self.criterion = torch.nn.NLLLoss(reduction='none') if self.is_prob else torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.is_prob:
            scores = torch.clamp(input, min=self.eps, max=1 - self.eps).to(input.device)
            ce_loss = self.criterion(torch.log(scores), target)
        else:
            scores = torch.clamp(F.softmax(input, dim=1), min=self.eps, max=1 - self.eps).to(input.device)
            ce_loss = self.criterion(input, target)
        confidence = self.score_func(scores)
        with torch.no_grad():
            indices_sorted = torch.argsort(confidence, descending=False)
            reverse_indices = torch.argsort(indices_sorted).to(dtype=torch.long)
            reordered_alphas = torch.tensor(self.alphas, dtype=input.dtype, device=input.device)[reverse_indices]
        losses = ce_loss * reordered_alphas
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses



class AURCLoss(BaseAURCLoss):
    def __init__(self, batch_size=128, score_function="MSP", reduction='mean', is_prob=False):
        super().__init__(batch_size=batch_size, score_function=score_function, reduction=reduction,
                         alpha_fn=compute_harmonic_alphas, is_prob=is_prob)

