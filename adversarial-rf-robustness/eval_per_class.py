"""
Per-modulation vulnerability analysis.

Evaluates adversarial robustness broken down by modulation class
to identify which schemes are most/least vulnerable to attack.

Usage:
  python eval_per_class.py --model_path experiments/results/baseline_cnn/best_model.pth
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.dataset import RadioMLDataset, get_dataloaders
from models.cnn import RFClassifierCNN
from channels.awgn import AWGNChannel
from channels.composite import CompositeChannel
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from utils.metrics import compute_per_class_prf


# Modulation labels by dataset version
MOD_CLASSES_2016 = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
    "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"
]

MOD_CLASSES_2018 = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC",
    "AM-DSB-SC", "FM", "GMSK", "OQPSK"
]

def get_mod_classes(version="2016.10a"):
    return MOD_CLASSES_2016 if version == "2016.10a" else MOD_CLASSES_2018

# Default for backward compatibility
MOD_CLASSES = MOD_CLASSES_2016


def get_device(device_str="auto"):
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def eval_per_class(model, test_loader, device, snr_db=10.0, rho=0.01,
                   attack_type="pgd", pgd_steps=10, num_trials=5):
    """
    Per-class clean accuracy, robust accuracy, and ASR at a fixed SNR.

    Methodology: Samples are filtered by native dataset SNR label (no double-noising).
    No additional AWGN is applied since RadioML samples already have noise at
    their labeled SNR level.

    Returns DataFrame with columns:
      modulation, clean_acc, robust_acc, asr, clean_acc_std, robust_acc_std, asr_std, n_samples
    """
    num_classes = len(MOD_CLASSES)
    attack_fn = pgd_attack if attack_type == "pgd" else fgsm_attack

    # Accumulate per-trial, per-class results
    trial_data = {c: {"clean": [], "robust": [], "fooled": [], "total": []}
                  for c in range(num_classes)}

    for trial in range(num_trials):
        class_correct_clean = [0] * num_classes
        class_correct_adv = [0] * num_classes
        class_fooled = [0] * num_classes
        class_total = [0] * num_classes

        for batch in test_loader:
            x = batch["iq"].to(device)
            y = batch["label"].to(device)

            # Filter to samples at target SNR (within 1 dB)
            snr = batch["snr"].float()
            mask = (snr >= snr_db - 1.0) & (snr <= snr_db + 1.0)
            if mask.sum() == 0:
                continue
            x, y = x[mask], y[mask]

            # Clean prediction -- native samples, no additional AWGN
            with torch.no_grad():
                preds_clean = model(x).argmax(1)

            # Adversarial attack -- no channel (AWGN already in data)
            attack_kwargs = {"rho": rho}
            if attack_type == "pgd":
                attack_kwargs["num_steps"] = pgd_steps
            x_adv = attack_fn(model, x, y, **attack_kwargs)

            # Adversarial prediction -- native adversarial samples
            with torch.no_grad():
                preds_adv = model(x_adv).argmax(1)

            # Per-class accumulation
            for c in range(num_classes):
                c_mask = (y == c)
                if c_mask.sum() == 0:
                    continue
                n = c_mask.sum().item()
                class_total[c] += n
                class_correct_clean[c] += (preds_clean[c_mask] == y[c_mask]).sum().item()
                class_correct_adv[c] += (preds_adv[c_mask] == y[c_mask]).sum().item()
                # Fooled: correctly classified clean but misclassified after attack
                clean_correct = (preds_clean[c_mask] == y[c_mask])
                adv_wrong = (preds_adv[c_mask] != y[c_mask])
                class_fooled[c] += (clean_correct & adv_wrong).sum().item()

        for c in range(num_classes):
            if class_total[c] > 0:
                trial_data[c]["clean"].append(class_correct_clean[c] / class_total[c])
                trial_data[c]["robust"].append(class_correct_adv[c] / class_total[c])
                trial_data[c]["fooled"].append(
                    class_fooled[c] / max(class_correct_clean[c], 1))
                trial_data[c]["total"].append(class_total[c])

    # Aggregate with 95% confidence intervals
    rows = []
    for c in range(num_classes):
        n_trials = len(trial_data[c]["clean"])
        if n_trials > 0:
            # 95% CI using t-distribution
            ci_mult = stats.t.ppf(0.975, df=max(n_trials - 1, 1)) if n_trials > 1 else 0
            clean_std = np.std(trial_data[c]["clean"], ddof=1) if n_trials > 1 else 0
            robust_std = np.std(trial_data[c]["robust"], ddof=1) if n_trials > 1 else 0
            asr_std = np.std(trial_data[c]["fooled"], ddof=1) if n_trials > 1 else 0
            se_clean = clean_std / np.sqrt(n_trials)
            se_robust = robust_std / np.sqrt(n_trials)
            se_asr = asr_std / np.sqrt(n_trials)

            rows.append({
                "modulation": MOD_CLASSES[c],
                "clean_acc": np.mean(trial_data[c]["clean"]),
                "robust_acc": np.mean(trial_data[c]["robust"]),
                "asr": np.mean(trial_data[c]["fooled"]),
                "clean_acc_std": clean_std,
                "robust_acc_std": robust_std,
                "asr_std": asr_std,
                "clean_ci95": ci_mult * se_clean,
                "robust_ci95": ci_mult * se_robust,
                "asr_ci95": ci_mult * se_asr,
                "n_samples": int(np.mean(trial_data[c]["total"])),
                "n_trials": n_trials,
            })

    df = pd.DataFrame(rows)
    return df


def eval_confusion_matrix(model, test_loader, device, snr_db=10.0, rho=0.01,
                          attack_type="pgd", pgd_steps=10):
    """
    Generate confusion matrices for clean and adversarial predictions.

    Samples filtered by native SNR label; no additional AWGN applied.
    """
    num_classes = len(MOD_CLASSES)
    attack_fn = pgd_attack if attack_type == "pgd" else fgsm_attack

    clean_cm = np.zeros((num_classes, num_classes), dtype=int)
    adv_cm = np.zeros((num_classes, num_classes), dtype=int)

    for batch in test_loader:
        x = batch["iq"].to(device)
        y = batch["label"].to(device)

        snr = batch["snr"].float()
        mask = (snr >= snr_db - 1.0) & (snr <= snr_db + 1.0)
        if mask.sum() == 0:
            continue
        x, y = x[mask], y[mask]

        # Clean prediction -- native samples, no double-noising
        with torch.no_grad():
            preds_clean = model(x).argmax(1)

        attack_kwargs = {"rho": rho}
        if attack_type == "pgd":
            attack_kwargs["num_steps"] = pgd_steps
        x_adv = attack_fn(model, x, y, **attack_kwargs)
        with torch.no_grad():
            preds_adv = model(x_adv).argmax(1)

        for i in range(y.size(0)):
            clean_cm[y[i].item()][preds_clean[i].item()] += 1
            adv_cm[y[i].item()][preds_adv[i].item()] += 1

    return clean_cm, adv_cm


def eval_per_class_prf(model, test_loader, device, snr_db=10.0, rho=0.01,
                       attack_type="pgd", pgd_steps=10):
    """
    Collect all predictions at a given SNR and compute per-class
    precision, recall, and F1 for both clean and adversarial settings.

    Returns:
        clean_df, clean_report, adv_df, adv_report
    """
    attack_fn = pgd_attack if attack_type == "pgd" else fgsm_attack

    all_y_true, all_clean_pred, all_adv_pred = [], [], []

    for batch in test_loader:
        x = batch["iq"].to(device)
        y = batch["label"].to(device)

        snr = batch["snr"].float()
        mask = (snr >= snr_db - 1.0) & (snr <= snr_db + 1.0)
        if mask.sum() == 0:
            continue
        x, y = x[mask], y[mask]

        with torch.no_grad():
            preds_clean = model(x).argmax(1)

        attack_kwargs = {"rho": rho}
        if attack_type == "pgd":
            attack_kwargs["num_steps"] = pgd_steps
        x_adv = attack_fn(model, x, y, **attack_kwargs)
        with torch.no_grad():
            preds_adv = model(x_adv).argmax(1)

        all_y_true.append(y.cpu().numpy())
        all_clean_pred.append(preds_clean.cpu().numpy())
        all_adv_pred.append(preds_adv.cpu().numpy())

    y_true = np.concatenate(all_y_true)
    clean_pred = np.concatenate(all_clean_pred)
    adv_pred = np.concatenate(all_adv_pred)

    clean_df, clean_report = compute_per_class_prf(y_true, clean_pred, MOD_CLASSES)
    adv_df, adv_report = compute_per_class_prf(y_true, adv_pred, MOD_CLASSES)

    return clean_df, clean_report, adv_df, adv_report


def plot_confusion_matrix(cm, class_names, title, save_path):
    """
    Generate a publication-quality greyscale heatmap from a confusion matrix.
    """
    # Normalize rows to percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Greys", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-Class Vulnerability Analysis")
    parser.add_argument("--model_path", type=str,
                        default="experiments/results/baseline_cnn/best_model.pth")
    parser.add_argument("--data_path", type=str,
                        default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--dataset_version", type=str, default="2016.10a",
                        choices=["2016.10a", "2018.01a"])
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--input_length", type=int, default=128)
    parser.add_argument("--snr", type=float, nargs="+", default=[0, 10, 18])
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--attack", type=str, default="pgd", choices=["pgd", "fgsm"])
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # Load model
    model = RFClassifierCNN(num_classes=args.num_classes, input_length=args.input_length).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Set MOD_CLASSES based on dataset version
    global MOD_CLASSES
    MOD_CLASSES = get_mod_classes(args.dataset_version)

    # Load data
    _, _, test_loader = get_dataloaders(
        data_path=args.data_path, dataset_version=args.dataset_version,
        snr_range=None, batch_size=256, num_workers=0, seed=42)  # Load all SNRs; filtering done in eval
    print(f"Test set: {len(test_loader.dataset):,} samples")

    # Per-class analysis at each SNR
    all_dfs = []
    for snr in args.snr:
        print(f"\n{'='*60}")
        print(f"Per-Class Analysis at SNR = {snr} dB ({args.attack.upper()}, rho={args.rho})")
        print(f"{'='*60}")

        df = eval_per_class(model, test_loader, device,
                            snr_db=snr, rho=args.rho,
                            attack_type=args.attack, pgd_steps=args.pgd_steps,
                            num_trials=args.num_trials)
        df["snr"] = snr
        all_dfs.append(df)

        # Pretty print with 95% CI
        print(f"\n{'Modulation':<12} {'Clean':>14} {'Robust':>14} {'ASR':>14} {'N':>6}")
        print("-" * 66)
        for _, row in df.iterrows():
            print(f"{row['modulation']:<12} "
                  f"{row['clean_acc']:>5.1%}+/-{row['clean_ci95']:.1%} "
                  f"{row['robust_acc']:>5.1%}+/-{row['robust_ci95']:.1%} "
                  f"{row['asr']:>5.1%}+/-{row['asr_ci95']:.1%} "
                  f"{row['n_samples']:>6d}")

        # Most/least vulnerable
        if len(df) > 0:
            most_vuln = df.loc[df["asr"].idxmax()]
            least_vuln = df.loc[df["asr"].idxmin()]
            print(f"\n  Most vulnerable:  {most_vuln['modulation']} (ASR={most_vuln['asr']:.1%})")
            print(f"  Least vulnerable: {least_vuln['modulation']} (ASR={least_vuln['asr']:.1%})")

    # Save combined results
    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(args.output_dir, "per_class_vulnerability.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Confusion matrix at SNR=10
    print(f"\n{'='*60}")
    print("Generating confusion matrices at SNR=10 dB...")
    clean_cm, adv_cm = eval_confusion_matrix(
        model, test_loader, device, snr_db=10.0,
        rho=args.rho, attack_type=args.attack, pgd_steps=args.pgd_steps)

    cm_path = os.path.join(args.output_dir, "confusion_matrices.npz")
    np.savez(cm_path, clean=clean_cm, adversarial=adv_cm, classes=MOD_CLASSES)
    print(f"Confusion matrices saved to {cm_path}")

    # Confusion matrix heatmaps
    fig_dir = args.output_dir.replace("results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_confusion_matrix(
        clean_cm, MOD_CLASSES,
        "Clean Confusion Matrix (SNR=10 dB)",
        os.path.join(fig_dir, "confusion_matrix_clean.png"))
    plot_confusion_matrix(
        adv_cm, MOD_CLASSES,
        f"Adversarial Confusion Matrix (SNR=10 dB, {args.attack.upper()} ρ={args.rho})",
        os.path.join(fig_dir, "confusion_matrix_adv.png"))

    # Per-class precision / recall / F1
    print(f"\n{'='*60}")
    print("Per-Class Precision / Recall / F1 at SNR=10 dB")
    print(f"{'='*60}")
    clean_prf, clean_report, adv_prf, adv_report = eval_per_class_prf(
        model, test_loader, device, snr_db=10.0,
        rho=args.rho, attack_type=args.attack, pgd_steps=args.pgd_steps)

    print("\n--- Clean ---")
    print(clean_report)
    print("\n--- Adversarial ---")
    print(adv_report)

    clean_prf.to_csv(os.path.join(args.output_dir, "per_class_prf_clean.csv"), index=False)
    adv_prf.to_csv(os.path.join(args.output_dir, "per_class_prf_adv.csv"), index=False)
    print(f"PRF tables saved to {args.output_dir}/per_class_prf_*.csv")


if __name__ == "__main__":
    main()
