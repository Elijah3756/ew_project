"""
Fast Gradient Sign Method (FGSM) adapted for I/Q signals
with power-ratio perturbation constraint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    rho: float = 0.01,
    channel: nn.Module = None,
    snr_db: float = None,
    targeted: bool = False,
    target_label: torch.Tensor = None,
) -> torch.Tensor:
    """
    FGSM attack with L2 power-ratio constraint.

    Perturbation budget: ||delta||_2 / ||x||_2 <= rho

    Args:
        model: Target classifier.
        x: Clean I/Q input, shape (batch, 2, T).
        y: True labels, shape (batch,).
        rho: Maximum perturbation power ratio.
        channel: Optional channel layer applied after perturbation.
        snr_db: SNR for channel (if applicable).
        targeted: If True, minimize loss toward target_label.
        target_label: Required if targeted=True.
    Returns:
        Adversarial examples x_adv, same shape as x.
    """
    x_adv = x.clone().detach().requires_grad_(True)

    # Forward pass (optionally through channel)
    if channel is not None:
        logits = model(channel(x_adv, snr_db=snr_db) if snr_db else channel(x_adv))
    else:
        logits = model(x_adv)

    # Compute loss
    if targeted:
        loss = -F.cross_entropy(logits, target_label)
    else:
        loss = F.cross_entropy(logits, y)

    # Backward
    loss.backward()

    # Gradient direction (L2 normalized)
    grad = x_adv.grad.detach()
    grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1, keepdim=True)
    grad_norm = grad_norm.unsqueeze(-1)  # (batch, 1, 1)
    grad_normalized = grad / (grad_norm + 1e-12)

    # Scale to power-ratio budget
    x_norm = torch.norm(x.view(x.shape[0], -1), p=2, dim=1, keepdim=True).unsqueeze(-1)
    epsilon = rho * x_norm

    # Perturbation
    delta = epsilon * grad_normalized

    x_adv = (x + delta).detach()

    return x_adv
