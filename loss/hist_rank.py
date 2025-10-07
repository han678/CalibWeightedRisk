"""
Differentiable, bin-based (histogram) CDF -> approximate rank.
Simplified version without edge update functionality.
"""

from typing import Optional
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class DifferentiableHistogramCDF:
    """
    Bin-based, differentiable CDF using smoothed Heaviside at bin centers.
    """
    def __init__(self, edges: torch.Tensor, sigma: float = 0.05, device: Optional[torch.device] = None, 
                 dtype: torch.dtype = torch.float32):
        assert edges.dim() == 1 and edges.numel() >= 2, "edges must be 1D with length >= 2"
        self.device = device or edges.device
        self.dtype = dtype
        self.edges = edges.to(self.device, self.dtype)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.K = self.centers.numel()
        self.sigma = float(sigma)
        self.counts = torch.zeros(self.K, device=self.device, dtype=self.dtype)
        self.total = torch.tensor(0.0, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def clear(self):
        """Clear histogram counts"""
        self.counts.zero_()
        self.total.zero_()

    @torch.no_grad()
    def add(self, scores: torch.Tensor, weight: float = 1.0):
        """Add scores to histogram bins"""
        s = scores.detach().to(self.device, self.dtype).flatten()
        idx = torch.bucketize(s, self.edges, right=True) - 1
        idx = idx.clamp(0, self.K - 1)
        binc = torch.bincount(idx, minlength=self.K).to(self.device, self.dtype)
        if weight != 1.0:
            binc *= float(weight)
        self.counts += binc
        self.total += float(s.numel()) * float(weight)

    @torch.no_grad()
    def decay(self, factor: float):
        """Exponential decay of history"""
        factor = float(factor)
        assert 0.0 < factor <= 1.0
        self.counts.mul_(factor)
        self.total.mul_(factor)

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Compute CDF values for input x"""
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(self.device, self.dtype)
        x = x.view(-1, 1)
        centers = self.centers.view(1, -1)
        weights = self.counts.clamp_min(1e-12).view(1, -1)
        S = torch.sigmoid((x - centers) / self.sigma)
        num = (S * weights).sum(dim=1)
        den = self.total.clamp_min(1e-12)
        return num / den

    def approx_rank(self, x: torch.Tensor) -> torch.Tensor:
        """Compute approximate rank"""
        p = self.cdf(x)
        N = self.total.clamp_min(1.0)
        return p * N + 1.0

    @property
    def size(self) -> int:
        """Return total number of samples"""
        return int(self.total.item())

# Simple test function
def test_sigma_effect():
    """Test CDF with different sigma values"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate test data
    np.random.seed(0)
    data = np.random.normal(loc=0, scale=2, size=1000)
    edges = torch.linspace(-8, 8, 65, device=device)
    test_x = torch.linspace(-8, 8, 200, device=device)

    # Empirical CDF
    data_sorted = np.sort(data)
    emp_cdf = np.searchsorted(data_sorted, test_x.cpu().numpy(), side='right') / len(data)

    plt.figure(figsize=(10, 6))
    plt.plot(test_x.cpu().numpy(), emp_cdf, label='Empirical CDF', color='black', linewidth=2)

    # Test different sigma values
    for sigma in [0.01, 0.05, 0.1, 0.5, 1.0]:
        hist = DifferentiableHistogramCDF(edges, sigma=sigma, device=device)
        hist.add(torch.from_numpy(data).to(device))
        cdf_vals = hist.cdf(test_x).cpu().numpy()
        plt.plot(test_x.cpu().numpy(), cdf_vals, label=f'sigma={sigma}')

    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.title('Differentiable Histogram CDF vs Empirical CDF')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_sigma_effect()