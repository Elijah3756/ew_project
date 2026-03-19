"""
Publication-ready figure generation for adversarial RF robustness paper.

Reads CSV results from experiments/results/ and produces figures
for experiments/figures/.

Usage:
  python plot_results.py --results_dir experiments/results --output_dir experiments/figures
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Publication style — grayscale-compatible for CDR print
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6.5, 4.5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})

# Grayscale palette — distinguishable in both print and screen
GRAYS = ["#000000", "#555555", "#999999", "#BBBBBB", "#333333"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p"]
HATCHES = ["///", "\\\\\\", "xxx", "...", "|||", "---"]

CHANNEL_LABELS = {
    "awgn": "AWGN",
    "rayleigh_awgn": "Rayleigh + AWGN",
    "rayleigh_cfo_awgn": "Rayleigh + CFO + AWGN",
}

DEFENSE_LABELS = {
    "baseline": "Baseline",
    "channel_aug": "Channel Aug.",
    "adv_train": "Adv. Training",
    "noise_inject": "Noise Injection",
    "adv_train_channel": "Adv.+Channel",
}


# ============================================================
# Figure 1: Clean Accuracy vs SNR (multiple channels)
# ============================================================
def plot_clean_accuracy_vs_snr(results_dir, output_dir):
    """Plot clean accuracy vs SNR for different channel conditions."""
    fig, ax = plt.subplots()

    files = sorted(glob.glob(os.path.join(results_dir, "*/clean_snr_*.csv"))) + \
            sorted(glob.glob(os.path.join(results_dir, "clean_snr_*.csv")))

    if not files:
        print("  No clean_snr CSV files found. Skipping.")
        return

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        ch_type = fname.replace("clean_snr_", "").replace(".csv", "")
        label = CHANNEL_LABELS.get(ch_type, ch_type)

        df = pd.read_csv(fpath)
        ax.plot(df["snr"], df["accuracy"] * 100,
                marker=MARKERS[i % len(MARKERS)],
                color=GRAYS[i % len(GRAYS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                label=label, markevery=2)

        if "accuracy_std" in df.columns:
            ax.fill_between(
                df["snr"],
                (df["accuracy"] - df["accuracy_std"]) * 100,
                (df["accuracy"] + df["accuracy_std"]) * 100,
                alpha=0.15, color=GRAYS[i % len(GRAYS)],
            )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title("Clean Accuracy vs SNR Under Channel Impairments")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    ax.set_xlim(-10, 18)

    save_path = os.path.join(output_dir, "fig1_clean_accuracy_vs_snr.pdf")
    fig.savefig(save_path)
    fig.savefig(save_path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Figure 2: Robust Accuracy vs SNR (attack comparison)
# ============================================================
def plot_robust_accuracy_vs_snr(results_dir, output_dir):
    """Plot robust accuracy under different attacks vs SNR."""
    fig, ax = plt.subplots()

    files = sorted(glob.glob(os.path.join(results_dir, "*/attack_snr_*.csv"))) + \
            sorted(glob.glob(os.path.join(results_dir, "attack_snr_*.csv")))

    if not files:
        print("  No attack_snr CSV files found. Skipping.")
        return

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        df = pd.read_csv(fpath)

        # Parse attack info from filename
        parts = fname.replace("attack_snr_", "").replace(".csv", "").split("_")
        attack = parts[0].upper() if parts else "?"
        channel = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1] if len(parts) > 1 else "?"
        rho_str = parts[-1] if parts[-1].startswith("rho") else ""

        label = f"{attack} ({CHANNEL_LABELS.get(channel, channel)})"
        if rho_str:
            rho_val = float(rho_str.replace("rho", ""))
            label += f" rho={rho_val*100:.1f}%"

        # Plot clean accuracy
        if i == 0 and "clean_acc" in df.columns:
            ax.plot(df["snr"], df["clean_acc"] * 100,
                    marker="o", color=GRAYS[0], label="Clean (no attack)",
                    linestyle="--", markevery=2)

        # Plot robust accuracy
        if "robust_acc" in df.columns:
            ax.plot(df["snr"], df["robust_acc"] * 100,
                    marker=MARKERS[(i + 1) % len(MARKERS)],
                    color=GRAYS[(i + 1) % len(GRAYS)],
                    linestyle=LINESTYLES[(i + 1) % len(LINESTYLES)],
                    label=label, markevery=2)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robust Accuracy vs SNR Under Adversarial Attack")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_xlim(-10, 18)

    save_path = os.path.join(output_dir, "fig2_robust_accuracy_vs_snr.pdf")
    fig.savefig(save_path)
    fig.savefig(save_path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Figure 3: Attack Success Rate vs Perturbation Budget
# ============================================================
def plot_attack_vs_budget(results_dir, output_dir):
    """Plot attack success rate vs perturbation budget at fixed SNRs."""
    files = sorted(glob.glob(os.path.join(results_dir, "*/attack_budget_*.csv"))) + \
            sorted(glob.glob(os.path.join(results_dir, "attack_budget_*.csv")))

    if not files:
        print("  No attack_budget CSV files found. Skipping.")
        return

    for fpath in files:
        df = pd.read_csv(fpath)
        snr_values = sorted(df["snr"].unique())

        fig, ax = plt.subplots()

        for i, snr in enumerate(snr_values):
            sub = df[df["snr"] == snr].sort_values("rho")
            ax.plot(sub["rho"] * 100, sub["attack_success_rate"] * 100,
                    marker=MARKERS[i % len(MARKERS)],
                    color=GRAYS[i % len(GRAYS)],
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    label=f"SNR = {snr} dB")

            if "asr_std" in sub.columns:
                ax.fill_between(
                    sub["rho"] * 100,
                    (sub["attack_success_rate"] - sub["asr_std"]) * 100,
                    (sub["attack_success_rate"] + sub["asr_std"]) * 100,
                    alpha=0.15, color=GRAYS[i % len(GRAYS)],
                )

        fname = os.path.basename(fpath)
        attack = fname.split("_")[2].upper()

        ax.set_xlabel("Perturbation Budget $\\rho$ (%)")
        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_title(f"{attack} Attack Success vs Perturbation Budget")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 105)

        save_path = os.path.join(output_dir, f"fig3_attack_budget_{attack.lower()}.pdf")
        fig.savefig(save_path)
        fig.savefig(save_path.replace(".pdf", ".png"))
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ============================================================
# Figure 4: Defense Comparison (grouped bar chart)
# ============================================================
def plot_defense_comparison(results_dir, output_dir):
    """
    Plot defense comparison: clean acc vs robust acc for each defense.
    Expects CSV files named defense_comparison_*.csv with columns:
    defense, snr, clean_acc, robust_acc
    """
    files = sorted(glob.glob(os.path.join(results_dir, "*/defense_comparison*.csv"))) + \
            sorted(glob.glob(os.path.join(results_dir, "defense_comparison*.csv")))

    if not files:
        print("  No defense_comparison CSV files found. Skipping.")
        return

    for fpath in files:
        df = pd.read_csv(fpath)
        defenses = df["defense"].unique()
        snr_values = sorted(df["snr"].unique())

        fig, axes = plt.subplots(1, len(snr_values), figsize=(5 * len(snr_values), 5), sharey=True)
        if len(snr_values) == 1:
            axes = [axes]

        for ax, snr in zip(axes, snr_values):
            sub = df[df["snr"] == snr]
            x = np.arange(len(sub))
            width = 0.35

            bars1 = ax.bar(x - width / 2, sub["clean_acc"].values * 100, width,
                           label="Clean Acc", color="#CCCCCC", edgecolor="black",
                           linewidth=0.8, hatch="")
            bars2 = ax.bar(x + width / 2, sub["robust_acc"].values * 100, width,
                           label="Robust Acc", color="#666666", edgecolor="black",
                           linewidth=0.8, hatch="///")

            ax.set_xlabel("Defense")
            ax.set_ylabel("Accuracy (%)" if snr == snr_values[0] else "")
            ax.set_title(f"SNR = {snr} dB")
            ax.set_xticks(x)
            labels = [DEFENSE_LABELS.get(d, d) for d in sub["defense"]]
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylim(0, 105)

        fig.suptitle("Defense Comparison: Clean vs Robust Accuracy", fontsize=14)
        fig.tight_layout()

        save_path = os.path.join(output_dir, "fig4_defense_comparison.pdf")
        fig.savefig(save_path)
        fig.savefig(save_path.replace(".pdf", ".png"))
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ============================================================
# Figure 5: Training History
# ============================================================
def plot_training_history(results_dir, output_dir):
    """Plot training loss and accuracy curves."""
    files = sorted(glob.glob(os.path.join(results_dir, "*/training_history.csv")))

    if not files:
        print("  No training_history CSV files found. Skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for fpath in files:
        label = os.path.basename(os.path.dirname(fpath)).replace("_", " ").title()
        df = pd.read_csv(fpath)
        epochs = range(1, len(df) + 1)

        # Plot training loss (all formats have train_loss)
        ax1.plot(epochs, df["train_loss"], label=f"{label} (train)", linestyle="-")
        if "val_loss" in df.columns:
            ax1.plot(epochs, df["val_loss"], label=f"{label} (val)", linestyle="--")

        # Handle different column names for train accuracy
        train_acc_col = "train_acc" if "train_acc" in df.columns else "train_acc_clean"
        ax2.plot(epochs, [a * 100 for a in df[train_acc_col]], label=f"{label} (train)", linestyle="-")
        ax2.plot(epochs, [a * 100 for a in df["val_acc"]], label=f"{label} (val)", linestyle="--")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training / Validation Loss")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training / Validation Accuracy")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "fig0_training_history.pdf")
    fig.savefig(save_path)
    fig.savefig(save_path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--results_dir", type=str, default="experiments/results")
    parser.add_argument("--output_dir", type=str, default="experiments/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating publication figures...")
    print("-" * 50)

    print("\n[Fig 0] Training History")
    plot_training_history(args.results_dir, args.output_dir)

    print("\n[Fig 1] Clean Accuracy vs SNR")
    plot_clean_accuracy_vs_snr(args.results_dir, args.output_dir)

    print("\n[Fig 2] Robust Accuracy vs SNR")
    plot_robust_accuracy_vs_snr(args.results_dir, args.output_dir)

    print("\n[Fig 3] Attack Success vs Perturbation Budget")
    plot_attack_vs_budget(args.results_dir, args.output_dir)

    print("\n[Fig 4] Defense Comparison")
    plot_defense_comparison(args.results_dir, args.output_dir)

    print("\n" + "-" * 50)
    print(f"All figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
