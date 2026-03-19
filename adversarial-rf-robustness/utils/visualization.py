"""
Publication-ready plotting utilities.
"""

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from typing import List, Optional

# Publication style
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6, 4),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_accuracy_vs_snr(
    results: dict,
    title: str = "Classification Accuracy vs SNR",
    save_path: Optional[str] = None,
):
    """
    Plot accuracy vs SNR for multiple configurations.

    Args:
        results: Dict of {label: DataFrame with 'snr' and 'accuracy' columns}.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots()

    markers = ["o", "s", "^", "D", "v", "<", ">"]
    for i, (label, df) in enumerate(results.items()):
        ax.plot(
            df["snr"], df["accuracy"] * 100,
            marker=markers[i % len(markers)],
            label=label, linewidth=1.5, markersize=5,
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_robust_accuracy_vs_snr(
    results: dict,
    title: str = "Robust Accuracy vs SNR",
    save_path: Optional[str] = None,
):
    """
    Plot robust accuracy (under attack) vs SNR.

    Args:
        results: Dict of {label: DataFrame with 'snr' and 'robust_accuracy'}.
    """
    fig, ax = plt.subplots()

    markers = ["o", "s", "^", "D", "v"]
    for i, (label, df) in enumerate(results.items()):
        ax.plot(
            df["snr"], df["robust_accuracy"] * 100,
            marker=markers[i % len(markers)],
            label=label, linewidth=1.5, markersize=5,
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Robust Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_attack_success_vs_budget(
    results: dict,
    title: str = "Attack Success Rate vs Perturbation Budget",
    save_path: Optional[str] = None,
):
    """
    Plot attack success rate vs perturbation budget (rho).

    Args:
        results: Dict of {label: DataFrame with 'rho' and 'attack_success_rate'}.
    """
    fig, ax = plt.subplots()

    markers = ["o", "s", "^", "D"]
    for i, (label, df) in enumerate(results.items()):
        ax.plot(
            df["rho"] * 100, df["attack_success_rate"] * 100,
            marker=markers[i % len(markers)],
            label=label, linewidth=1.5, markersize=5,
        )

    ax.set_xlabel("Perturbation Budget (rho %)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig
