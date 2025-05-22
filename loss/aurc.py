import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import digamma
from torch import Tensor
from torch.nn.modules.loss import _Loss
from fast_soft_sort.pytorch_ops import soft_rank

def cross_entropy_loss(probs, targets):
    """Compute cross-entropy loss between prediction probabilities and targets."""
    eps = 1e-9  # Small value to avoid log(0)
    probs = np.clip(probs, eps, 1 - eps)
    ce_loss = -np.sum(targets * np.log(probs), axis=1)
    return ce_loss

def zero_one_loss(probs, targets):
    """
    Compute the 0-1 loss between predicted probabilities and one-hot target labels.
    """
    predicted_labels = np.argmax(probs, axis=1)
    true_labels = np.argmax(targets, axis=1)
    incorrect_predictions = (predicted_labels != true_labels).astype(np.float32)
    return incorrect_predictions

# Utility functions for alpha computation
def compute_ln_alphas(n):
    """Compute logarithmic alphas."""
    return [-np.log(1 - rank / (n + 1)) for rank in range(1, n + 1)]


def compute_harmonic_alphas(n, use_diagamma=True):
    """Compute harmonic alphas using digamma or cumulative sum."""
    if use_diagamma:
        return [digamma(n + 1) - digamma(n - rank + 1) for rank in range(1, n + 1)]
    else:
        return np.cumsum(1 / (n - np.arange(n)))

# Utility functions for scoring
def entropy(x):
    """Compute entropy of input tensor."""
    eps = 1e-9  # Small value to avoid log(0)
    x = torch.clamp(x, min=eps, max=1 - eps)  # Clamp to avoid numerical instability
    return -torch.sum(torch.log(x) * x, dim=1)


def inv_softmax(x, c=torch.log(torch.tensor(10))):
    """Inverse softmax transformation."""
    return torch.log(x) + c


def top12_margin(x):
    """Compute the margin between top-1 and top-2 probabilities."""
    values, _ = torch.topk(x, k=2, dim=-1)
    return values[:, 0] - values[:, 1] if x.ndim > 1 else values[0] - values[1]

def get_score_function(name: str, input_is_softmax: bool = False):
    """Retrieve the appropriate scoring function."""
    eps = 1e-9
    as_probs = (lambda x: x) if input_is_softmax else (lambda x: torch.clamp(F.softmax(x, dim=1), min=eps, max=1 - eps))
    as_logits = (lambda x: inv_softmax(x)) if input_is_softmax else (lambda x: x)

    mapping = {
        "MSP": lambda x: torch.max(as_probs(x), dim=1).values,
        "NegEntropy": lambda x: -entropy(as_logits(x)),
        "SoftmaxMargin": lambda x: top12_margin(as_probs(x)),
    }

    if name not in mapping:
        raise ValueError(f"Unknown score function: {name}")
    return mapping[name]


class BaseAURCLoss(_Loss):
    def __init__(self, score_function="MSP", gamma=0.5, batch_size=128, reduction='sum',
                 regularization_strength=0.05, alpha_fn=None, weight_grad=True, input_is_softmax=False):
        super().__init__(reduction=reduction)
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_grad = weight_grad
        self.input_is_softmax = input_is_softmax
        self.regularization_strength = regularization_strength
        self.score_func = get_score_function(score_function, self.input_is_softmax)
        self.alphas = alpha_fn(batch_size) if alpha_fn else None
        self.eps = 1e-9  # small value for clamping
        if self.input_is_softmax:
            self.criterion = torch.nn.NLLLoss(reduction='none')
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.input_is_softmax:
            # Clamp input directly
            scores = torch.clamp(input, min=self.eps, max=1 - self.eps).to(input.device)
            log_scores = torch.log(scores)
            ce_loss = self.criterion(log_scores, target)
            confidence = self.score_func(scores)
        else:
            # Clamp softmax(input)
            softmaxed = F.softmax(input, dim=1)
            scores = torch.clamp(softmaxed, min=self.eps, max=1 - self.eps).to(input.device)
            ce_loss = self.criterion(input, target)
            confidence = self.score_func(scores)

        if not self.weight_grad:
            with torch.no_grad():
                indices_sorted = torch.argsort(confidence, descending=False)
                reverse_indices = torch.argsort(indices_sorted)
                reordered_alphas = torch.tensor(self.alphas, dtype=input.dtype, device=input.device)[reverse_indices]
            if self.input_is_softmax:
                losses = self.criterion(log_scores, target) * reordered_alphas
            else:
                losses = ce_loss * reordered_alphas
        else:
            rank = soft_rank(confidence.unsqueeze(0), regularization="l2",
                             regularization_strength=self.regularization_strength).to(input.device)
            alphas = -torch.log(1 - rank.squeeze(0) / (self.batch_size + 1))
            if self.input_is_softmax:
                losses = self.criterion(log_scores, target) * alphas
            else:
                losses = ce_loss * alphas

        if self.reduction == 'mean':
            return (1 - self.gamma) * ce_loss.mean() + self.gamma * losses.mean()
        elif self.reduction == 'sum':
            return (1 - self.gamma) * ce_loss.sum() + self.gamma * losses.sum()
        else:
            return (1 - self.gamma) * ce_loss + self.gamma * losses


# AURC loss class
class AURCLoss(BaseAURCLoss):
    def __init__(self, gamma=0.3, batch_size=128, score_function="MSP", reduction='sum',
                 weight_grad=True, input_is_softmax=False, regularization_strength=0.02):
        super().__init__(gamma=gamma, batch_size=batch_size, score_function=score_function, reduction=reduction,
                         regularization_strength=regularization_strength, alpha_fn=compute_harmonic_alphas,
                         weight_grad=weight_grad, input_is_softmax=input_is_softmax)

