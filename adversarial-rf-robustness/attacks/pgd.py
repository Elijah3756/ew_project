"""
Projected Gradient Descent (PGD) attack adapted for I/Q signals
with power-ratio perturbation constraint.

Supports two attack modes for fading channels:
- freeze_channel=False (default): Each PGD step uses an independent channel realization
  (no-CSI stochastic attack, like an implicit Expectation over Transformation).
- freeze_channel=True: All PGD steps use the same fixed channel realization
  (perfect CSI attack, optimal if attacker has channel knowledge).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _project_l2(delta: torch.Tensor, x: torch.Tensor, rho: float) -> torch.Tensor:
    """Project perturbation onto L2 ball: ||delta||_2 <= rho * ||x||_2."""
    x_norm = torch.norm(x.view(x.shape[0], -1), p=2, dim=1, keepdim=True).unsqueeze(-1)
    delta_norm = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1, keepdim=True).unsqueeze(-1)
    max_norm = rho * x_norm
    factor = torch.clamp(max_norm / (delta_norm + 1e-12), max=1.0)
    return delta * factor


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    rho: float = 0.01,
    step_size: float = None,
    num_steps: int = 10,
    channel: nn.Module = None,
    snr_db: float = None,
    targeted: bool = False,
    target_label: torch.Tensor = None,
    random_start: bool = True,
    freeze_channel: bool = False,
) -> torch.Tensor:
    """
    PGD attack with L2 power-ratio constraint.

    Args:
        model: Target classifier.
        x: Clean I/Q input, shape (batch, 2, T).
        y: True labels, shape (batch,).
        rho: Maximum perturbation power ratio.
        step_size: Step size per iteration. Default: rho / 4.
        num_steps: Number of PGD iterations.
        channel: Optional channel layer applied after perturbation.
        snr_db: SNR for channel (if applicable).
        targeted: If True, minimize loss toward target_label.
        target_label: Required if targeted=True.
        random_start: If True, initialize with random perturbation.
        freeze_channel: If True, use a fixed channel realization across all PGD steps
            (perfect CSI attack). If False (default), each step sees an independent
            channel realization (no-CSI stochastic attack). Only used if channel is not None.
    Returns:
        Adversarial examples x_adv, same shape as x.
    """
    if step_size is None:
        step_size = rho / 4.0

    x_orig = x.clone().detach()

    # Random initialization within budget
    if random_start:
        delta = torch.randn_like(x)
        delta = _project_l2(delta, x_orig, rho)
    else:
        delta = torch.zeros_like(x)

    # Store RNG state and generate fixed channel seed if freeze_channel=True
    channel_seed = None
    if freeze_channel and channel is not None:
        # Generate and store a fixed random seed for the channel
        channel_seed = torch.randint(0, 2**31, (1,)).item()
        rng_state = torch.get_rng_state()

    for _ in range(num_steps):
        delta = delta.detach().requires_grad_(True)
        x_adv = x_orig + delta

        # Forward pass
        if channel is not None:
            # If freeze_channel=True, use the same random seed each iteration
            if freeze_channel and channel_seed is not None:
                torch.manual_seed(channel_seed)

            logits = model(channel(x_adv, snr_db=snr_db) if snr_db else channel(x_adv))
        else:
            logits = model(x_adv)

        # Loss
        if targeted:
            loss = -F.cross_entropy(logits, target_label)
        else:
            loss = F.cross_entropy(logits, y)

        loss.backward()

        # Gradient step (L2 normalized)
        grad = delta.grad.detach()
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1, keepdim=True).unsqueeze(-1)
        grad_normalized = grad / (grad_norm + 1e-12)

        # Step size scaled to signal power
        x_norm = torch.norm(x_orig.view(x_orig.shape[0], -1), p=2, dim=1, keepdim=True).unsqueeze(-1)
        step = step_size * x_norm * grad_normalized

        delta = (delta + step).detach()

        # Project back onto constraint set
        delta = _project_l2(delta, x_orig, rho)

    # Restore RNG state if we froze the channel
    if freeze_channel and channel is not None and channel_seed is not None:
        torch.set_rng_state(rng_state)

    x_adv = (x_orig + delta).detach()
    return x_adv
