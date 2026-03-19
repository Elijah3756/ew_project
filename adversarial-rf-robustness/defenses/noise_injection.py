"""
Noise injection / randomized smoothing defense.
Adds calibrated Gaussian noise at inference to smooth decision boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NoiseInjectionDefense(nn.Module):
    """
    Wraps a classifier with Gaussian noise injection at inference.

    Can be used for:
      1. Training-time augmentation (noise during training)
      2. Inference-time smoothing (average predictions over noisy copies)
    """

    def __init__(self, model: nn.Module, sigma: float = 0.1, num_samples: int = 1):
        """
        Args:
            model: Base classifier.
            sigma: Standard deviation of Gaussian noise (relative to signal std).
            num_samples: Number of noisy copies to average at inference.
                1 = single noise injection, >1 = Monte Carlo smoothing.
        """
        super().__init__()
        self.model = model
        self.sigma = sigma
        self.num_samples = num_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input I/Q tensor, shape (batch, 2, T).
        Returns:
            Logits (averaged if num_samples > 1).
        """
        if self.training or self.num_samples == 1:
            # Single noise injection
            noise = torch.randn_like(x) * self.sigma
            return self.model(x + noise)
        else:
            # Monte Carlo smoothing: average softmax probabilities, return logits
            prob_sum = torch.zeros(x.shape[0], self.model.num_classes, device=x.device)
            for _ in range(self.num_samples):
                noise = torch.randn_like(x) * self.sigma
                prob_sum += F.softmax(self.model(x + noise), dim=1)
            avg_probs = prob_sum / self.num_samples
            # Convert back to logit scale for consistent downstream use
            return torch.log(avg_probs + 1e-12)

    def __repr__(self):
        return f"NoiseInjectionDefense(sigma={self.sigma}, num_samples={self.num_samples})"
