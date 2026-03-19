"""
Composite channel that chains multiple impairment layers.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CompositeChannel(nn.Module):
    """
    Chains multiple channel layers in sequence.

    Typical order: Rayleigh fading -> CFO -> AWGN
    (Signal distortions first, then additive noise last.)
    """

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, snr_db: float = None) -> torch.Tensor:
        """
        Args:
            x: Input I/Q tensor, shape (batch, 2, T).
            snr_db: If provided, passed to AWGN layer.
        Returns:
            Impaired signal with same shape.
        """
        for layer in self.layers:
            if hasattr(layer, "snr_db") and snr_db is not None:
                x = layer(x, snr_db=snr_db)
            else:
                x = layer(x)
        return x

    def __repr__(self):
        layer_strs = ", ".join(repr(l) for l in self.layers)
        return f"CompositeChannel([{layer_strs}])"
