"""
Differentiable Carrier Frequency Offset (CFO) channel layer.
Applies a rotating phasor to simulate frequency offset.
"""

import torch
import torch.nn as nn
import math


class CFOChannel(nn.Module):
    """
    Carrier Frequency Offset channel.

    Applies exp(j * 2 * pi * delta_f * t) to the complex signal,
    where delta_f is the normalized frequency offset.

    Input shape: (batch, 2, T) where dim 1 = [I, Q].
    """

    def __init__(self, max_offset: float = 0.01):
        """
        Args:
            max_offset: Maximum normalized frequency offset.
                Offset is sampled uniformly from [-max_offset, max_offset].
        """
        super().__init__()
        self.max_offset = max_offset

    def forward(self, x: torch.Tensor, offset: float = None) -> torch.Tensor:
        """
        Args:
            x: Input I/Q tensor, shape (batch, 2, T).
            offset: Fixed offset. If None, sampled randomly.
        Returns:
            Signal with CFO applied, same shape.
        """
        batch, _, T = x.shape

        if offset is None:
            # Random offset per batch element
            delta_f = (2.0 * torch.rand(batch, 1, 1, device=x.device) - 1.0) * self.max_offset
        else:
            delta_f = torch.full((batch, 1, 1), offset, device=x.device)

        # Time indices
        t = torch.arange(T, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

        # Phase rotation: theta = 2 * pi * delta_f * t
        theta = 2.0 * math.pi * delta_f * t  # (batch, 1, T)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Complex multiplication by exp(j*theta)
        x_i = x[:, 0:1, :]
        x_q = x[:, 1:2, :]

        y_i = cos_theta * x_i - sin_theta * x_q
        y_q = sin_theta * x_i + cos_theta * x_q

        return torch.cat([y_i, y_q], dim=1)

    def __repr__(self):
        return f"CFOChannel(max_offset={self.max_offset})"
