"""
Differentiable flat Rayleigh fading channel layer.
Applies a random complex gain drawn from a Rayleigh distribution.
"""

import torch
import torch.nn as nn


class RayleighFadingChannel(nn.Module):
    """
    Flat Rayleigh fading channel.

    Multiplies the complex I/Q signal by a random complex gain
    h ~ CN(0, 1), simulating flat fading.

    Input shape: (batch, 2, T) where dim 1 = [I, Q].
    """

    def __init__(self, coherence_samples: int = None):
        """
        Args:
            coherence_samples: Number of samples over which fading is constant.
                If None, fading is constant over the entire window (block fading).
        """
        super().__init__()
        self.coherence_samples = coherence_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input I/Q tensor, shape (batch, 2, T).
        Returns:
            Faded signal with same shape.
        """
        batch, _, T = x.shape

        if self.coherence_samples is None or self.coherence_samples >= T:
            # Block fading: one complex gain per batch element
            # h = h_r + j*h_i, where h_r, h_i ~ N(0, 0.5)
            h_real = torch.randn(batch, 1, 1, device=x.device) * (0.5 ** 0.5)
            h_imag = torch.randn(batch, 1, 1, device=x.device) * (0.5 ** 0.5)
        else:
            # Time-varying fading with given coherence
            num_blocks = (T + self.coherence_samples - 1) // self.coherence_samples
            h_real = torch.randn(batch, 1, num_blocks, device=x.device) * (0.5 ** 0.5)
            h_imag = torch.randn(batch, 1, num_blocks, device=x.device) * (0.5 ** 0.5)
            # Repeat to match T
            h_real = h_real.repeat_interleave(self.coherence_samples, dim=2)[:, :, :T]
            h_imag = h_imag.repeat_interleave(self.coherence_samples, dim=2)[:, :, :T]

        # Complex multiplication: (h_r + j*h_i)(x_I + j*x_Q)
        x_i = x[:, 0:1, :]  # I component
        x_q = x[:, 1:2, :]  # Q component

        y_i = h_real * x_i - h_imag * x_q
        y_q = h_real * x_q + h_imag * x_i

        return torch.cat([y_i, y_q], dim=1)

    def __repr__(self):
        return f"RayleighFadingChannel(coherence_samples={self.coherence_samples})"
