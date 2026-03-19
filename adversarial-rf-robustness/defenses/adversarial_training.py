"""
Adversarial training defense using PGD-lite during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from attacks.pgd import pgd_attack


def adversarial_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    rho: float = 0.01,
    pgd_steps: int = 5,
    channel: nn.Module = None,
    snr_db: float = None,
    alpha: float = 0.5,
) -> dict:
    """
    One step of adversarial training.

    Loss = alpha * L(clean) + (1-alpha) * L(adversarial)

    Args:
        model: Classifier to train.
        optimizer: Optimizer.
        x: Clean batch, shape (batch, 2, T).
        y: Labels, shape (batch,).
        rho: Perturbation budget for PGD.
        pgd_steps: Number of PGD steps (lite = fewer steps).
        channel: Optional channel for attack.
        snr_db: SNR for channel.
        alpha: Weight for clean loss (1-alpha for adversarial).
    Returns:
        Dict with loss values.
    """
    model.train()

    # Generate adversarial examples (model in eval for attack)
    model.eval()
    with torch.no_grad():
        pass  # PGD handles gradients internally
    x_adv = pgd_attack(
        model, x, y, rho=rho, num_steps=pgd_steps,
        channel=channel, snr_db=snr_db
    )
    model.train()

    # Forward on clean
    if channel is not None:
        logits_clean = model(channel(x, snr_db=snr_db) if snr_db else channel(x))
        logits_adv = model(channel(x_adv, snr_db=snr_db) if snr_db else channel(x_adv))
    else:
        logits_clean = model(x)
        logits_adv = model(x_adv)

    loss_clean = F.cross_entropy(logits_clean, y)
    loss_adv = F.cross_entropy(logits_adv, y)
    loss = alpha * loss_clean + (1.0 - alpha) * loss_adv

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss_total": loss.item(),
        "loss_clean": loss_clean.item(),
        "loss_adv": loss_adv.item(),
    }
