import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import numpy as np

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
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += prop_in_bin * (torch.abs(avg_confidence_in_bin - accuracy_in_bin)) ** self.p
        return ece

def fast_ece_vectorized(logits, labels, n_bins=15):
    """Ultra-minimal ECE for training monitoring - trades precision for speed."""
    with torch.no_grad():
        # Skip softmax if logits are already normalized
        if logits.max() <= 1.0 and logits.min() >= 0.0:
            confidences, predictions = torch.max(logits, 1)
        else:
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, 1)
        
        if labels.ndim > 1:
            labels = labels.argmax(1)
        
        # Simple equal-width bins (fastest)
        bin_size = 1.0 / n_bins
        bin_idx = (confidences / bin_size).long().clamp(0, n_bins-1)
        
        # Super fast aggregation using advanced indexing
        accuracies = predictions.eq(labels).float()
        
        # Use bincount for fastest aggregation
        counts = torch.bincount(bin_idx, minlength=n_bins).float()
        acc_sums = torch.bincount(bin_idx, weights=accuracies, minlength=n_bins)
        conf_sums = torch.bincount(bin_idx, weights=confidences, minlength=n_bins)
        
        # Vectorized computation
        mask = counts > 0
        avg_acc = torch.where(mask, acc_sums / counts, torch.zeros_like(counts))
        avg_conf = torch.where(mask, conf_sums / counts, torch.zeros_like(counts))
        
        # ECE
        weights = counts / len(confidences)
        ece = (weights * torch.abs(avg_conf - avg_acc)).sum()
        
        return ece.item()

class Ece(nn.Module):
    """Ultra-fast ECE implementation with minimal CPU-GPU transfers."""
    
    def __init__(self, p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=False, prob=False, return_per_class=False):
        super(Ece, self).__init__()
        self.p = p
        self.n_bins = n_bins
        self.version = version
        self.adaptive_bins = adaptive_bins
        self.classwise = classwise
        self.prob = prob
        self.return_per_class = return_per_class
        
        # For non-adaptive bins, pre-register boundaries
        if not adaptive_bins:
            self.register_buffer('bin_boundaries', torch.linspace(0, 1, n_bins + 1))
    
    def _get_bins_gpu(self, confidences):
        """Get bin boundaries entirely on GPU."""
        with torch.no_grad():
            if self.adaptive_bins:
                # GPU-only adaptive binning using quantiles
                sorted_conf, _ = torch.sort(confidences)
                n_samples = len(sorted_conf)
                indices = torch.linspace(0, n_samples - 1, self.n_bins + 1, device=confidences.device).long()
                indices = torch.clamp(indices, 0, n_samples - 1)
                bin_boundaries = sorted_conf[indices]
                return bin_boundaries
            else:
                return self.bin_boundaries.to(confidences.device)
    
    def _vectorized_ece(self, confidences, accuracies):
        """Vectorized ECE computation - much faster than loops."""
        with torch.no_grad():
            device = confidences.device
            bin_boundaries = self._get_bins_gpu(confidences)
            
            # Ultra-fast binning with torch.bucketize
            bin_indices = torch.bucketize(confidences, bin_boundaries, right=True) - 1
            bin_indices = torch.clamp(bin_indices, 0, self.n_bins - 1)
            
            # Vectorized aggregation
            bin_counts = torch.bincount(bin_indices, minlength=self.n_bins).float()
            bin_acc_sums = torch.bincount(bin_indices, weights=accuracies, minlength=self.n_bins)
            bin_conf_sums = torch.bincount(bin_indices, weights=confidences, minlength=self.n_bins)
            
            # Safe division
            mask = bin_counts > 0
            bin_accuracies = torch.zeros_like(bin_counts)
            bin_avg_confs = torch.zeros_like(bin_counts)
            
            bin_accuracies[mask] = bin_acc_sums[mask] / bin_counts[mask]
            bin_avg_confs[mask] = bin_conf_sums[mask] / bin_counts[mask]
            
            # ECE computation
            if self.version == "our":
                # Handle "our" version if needed
                total_samples = len(confidences)
                ece = torch.sum(torch.abs(bin_conf_sums - bin_acc_sums).pow(self.p)) / total_samples
            else:
                prop_in_bin = bin_counts / len(confidences)
                ece = torch.sum(prop_in_bin * torch.abs(bin_avg_confs - bin_accuracies).pow(self.p))
            
            return ece
    
    def forward(self, *, logits, labels, **kwargs):
        """Fast forward pass."""
        with torch.no_grad():
            try:
                if logits.numel() == 0 or labels.numel() == 0:
                    return 0.0
                
                # Process inputs
                softmaxes = logits if self.prob else F.softmax(logits, dim=1)
                if labels.ndim > 1:
                    if labels.shape[1] > 1:
                        labels = torch.argmax(labels, dim=1)
                    else:
                        labels = labels.squeeze()
                
                if self.classwise:
                    return self._compute_classwise_ece(softmaxes, labels)
                else:
                    return self._compute_global_ece(softmaxes, labels)
                    
            except Exception as e:
                print(f"[WARNING] Fast ECE computation failed: {e}")
                return 0.0
    
    def _compute_global_ece(self, softmaxes, labels):
        """Fast global ECE."""
        with torch.no_grad():
            confidences, predictions = torch.max(softmaxes, 1)
            accuracies = predictions.eq(labels).float()
            ece = self._vectorized_ece(confidences, accuracies)
            return ece.item()
    
    def _compute_classwise_ece(self, softmaxes, labels):
        """Fast class-wise ECE with batch processing."""
        with torch.no_grad():
            num_classes = softmaxes.shape[1]
            device = softmaxes.device
            
            # Pre-compute one-hot labels for all classes at once
            labels_onehot = F.one_hot(labels, num_classes).float()  # [N, C]
            
            # Batch process all classes
            all_ece = torch.zeros(num_classes, device=device)
            valid_classes = 0
            
            # Get global bin boundaries once for efficiency
            if not self.adaptive_bins:
                bin_boundaries = self.bin_boundaries.to(device)
            
            for i in range(num_classes):
                class_mask = labels_onehot[:, i] > 0
                if class_mask.sum() > 0:
                    class_confidences = softmaxes[:, i]
                    class_labels = labels_onehot[:, i]
                    
                    # Use same bins for all classes if not adaptive
                    if self.adaptive_bins:
                        # Only compute adaptive bins for classes with samples
                        ece = self._vectorized_ece(class_confidences, class_labels)
                    else:
                        # Reuse global bins - much faster
                        bin_indices = torch.bucketize(class_confidences, bin_boundaries, right=True) - 1
                        bin_indices = torch.clamp(bin_indices, 0, self.n_bins - 1)
                        
                        bin_counts = torch.bincount(bin_indices, minlength=self.n_bins).float()
                        bin_acc_sums = torch.bincount(bin_indices, weights=class_labels, minlength=self.n_bins)
                        bin_conf_sums = torch.bincount(bin_indices, weights=class_confidences, minlength=self.n_bins)
                        
                        mask = bin_counts > 0
                        bin_accuracies = torch.zeros_like(bin_counts)
                        bin_avg_confs = torch.zeros_like(bin_counts)
                        
                        bin_accuracies[mask] = bin_acc_sums[mask] / bin_counts[mask]
                        bin_avg_confs[mask] = bin_conf_sums[mask] / bin_counts[mask]
                        
                        prop_in_bin = bin_counts / len(class_confidences)
                        ece = torch.sum(prop_in_bin * torch.abs(bin_avg_confs - bin_accuracies))
                    
                    all_ece[i] = ece
                    valid_classes += 1
            
            if valid_classes > 0:
                if self.return_per_class:
                    return all_ece
                else:
                    return (all_ece.sum() / valid_classes).item()
            else:
                return 0.0
