"""
Evaluation metrics for adversarial robustness experiments.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_attack_success_rate(
    logits_clean: torch.Tensor,
    logits_adv: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Attack success rate: fraction of correctly classified samples
    that become misclassified after attack.
    """
    preds_clean = logits_clean.argmax(dim=1)
    preds_adv = logits_adv.argmax(dim=1)

    # Only consider samples that were correctly classified before attack
    correct_mask = (preds_clean == labels)
    if correct_mask.sum() == 0:
        return 0.0

    fooled = (preds_adv != labels) & correct_mask
    return fooled.float().sum().item() / correct_mask.float().sum().item()


def compute_per_class_prf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Per-class precision, recall, F1, and support.

    Returns:
        df: DataFrame with columns [class, precision, recall, f1, support]
        report: sklearn classification_report string for logging
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    labels = class_names if class_names is not None else [str(i) for i in range(len(precision))]
    df = pd.DataFrame({
        "class": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })
    report = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0
    )
    return df, report


def evaluate_snr_sweep(
    model: torch.nn.Module,
    dataloader,
    snr_values: List[float],
    channel=None,
    attack_fn=None,
    attack_kwargs: dict = None,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Evaluate model across SNR values.

    Returns DataFrame with columns: snr, accuracy, [attack_success_rate].
    """
    model.eval()
    results = []

    for snr in snr_values:
        all_correct = 0
        all_total = 0
        all_asr = []

        for batch in dataloader:
            x = batch["iq"].to(device)
            y = batch["label"].to(device)

            # Apply channel
            if channel is not None:
                x_ch = channel(x, snr_db=snr)
            else:
                x_ch = x

            with torch.no_grad():
                logits_clean = model(x_ch)

            acc = compute_accuracy(logits_clean, y)
            all_correct += (logits_clean.argmax(1) == y).sum().item()
            all_total += y.shape[0]

            # Attack if specified
            if attack_fn is not None:
                kwargs = attack_kwargs or {}
                x_adv = attack_fn(model, x, y, channel=channel, snr_db=snr, **kwargs)
                if channel is not None:
                    x_adv_ch = channel(x_adv, snr_db=snr)
                else:
                    x_adv_ch = x_adv
                with torch.no_grad():
                    logits_adv = model(x_adv_ch)
                asr = compute_attack_success_rate(logits_clean, logits_adv, y)
                all_asr.append(asr)

        row = {
            "snr": snr,
            "accuracy": all_correct / all_total,
        }
        if attack_fn is not None:
            row["attack_success_rate"] = np.mean(all_asr)
            row["robust_accuracy"] = all_correct / all_total * (1 - np.mean(all_asr))

        results.append(row)

    return pd.DataFrame(results)
