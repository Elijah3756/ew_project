"""
Defense training script for Phase 3.

Trains models with:
  1. Adversarial training (PGD-lite)
  2. Noise injection during training
  3. Combined: channel augmentation + adversarial training

Then evaluates all defended models against PGD attack and
produces a defense comparison CSV.

Usage:
  python train_defenses.py --data_path data/raw/RML2016.10a_dict.pkl
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import get_dataloaders
from models.cnn import RFClassifierCNN
from channels.awgn import AWGNChannel
from channels.rayleigh import RayleighFadingChannel
from channels.cfo import CFOChannel
from channels.composite import CompositeChannel
from attacks.pgd import pgd_attack
from attacks.fgsm import fgsm_attack
from utils.progress import iter_batches, write_status


def get_device(device_str: str = "auto") -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Training loops for each defense
# ============================================================

def train_epoch_clean(model, loader, optimizer, criterion, device,
                      channel=None, snr_range=None):
    """Standard clean training (with optional channel augmentation)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        x = batch["iq"].to(device)
        y = batch["label"].to(device)

        if channel is not None and snr_range is not None:
            snr = np.random.uniform(snr_range[0], snr_range[1])
            x = channel(x, snr_db=snr)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def train_epoch_adversarial(model, loader, optimizer, criterion, device,
                            rho=0.01, pgd_steps=5, alpha=0.5,
                            channel=None, snr_range=None):
    """Adversarial training with PGD-lite."""
    model.train()
    total_loss, correct_clean, correct_adv, total = 0.0, 0, 0, 0

    for batch in loader:
        x = batch["iq"].to(device)
        y = batch["label"].to(device)

        snr = None
        if channel is not None and snr_range is not None:
            snr = np.random.uniform(snr_range[0], snr_range[1])

        # Generate adversarial examples
        model.eval()
        x_adv = pgd_attack(
            model, x, y, rho=rho, num_steps=pgd_steps,
            channel=channel, snr_db=snr
        )
        model.train()

        # NOTE: x contains native AWGN. The channel applied inside PGD computes
        # h*(x_clean+n) rather than h*x_clean+n. This is a known approximation
        # inherent to using RadioML data with baked-in noise for pre-channel attacks.

        # Apply channel to both clean and adversarial
        if channel is not None and snr is not None:
            x_ch = channel(x, snr_db=snr)
            x_adv_ch = channel(x_adv, snr_db=snr)
        else:
            x_ch = x
            x_adv_ch = x_adv

        # Combined loss
        logits_clean = model(x_ch)
        logits_adv = model(x_adv_ch)

        loss_clean = criterion(logits_clean, y)
        loss_adv = criterion(logits_adv, y)
        loss = alpha * loss_clean + (1.0 - alpha) * loss_adv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct_clean += (logits_clean.argmax(1) == y).sum().item()
        correct_adv += (logits_adv.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct_clean / total, correct_adv / total


def train_epoch_noise_injection(model, loader, optimizer, criterion, device,
                                sigma=0.1, channel=None, snr_range=None):
    """Training with Gaussian noise injection."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        x = batch["iq"].to(device)
        y = batch["label"].to(device)

        if channel is not None and snr_range is not None:
            snr = np.random.uniform(snr_range[0], snr_range[1])
            x = channel(x, snr_db=snr)

        # Add noise injection
        noise = torch.randn_like(x) * sigma
        x_noisy = x + noise

        optimizer.zero_grad()
        logits = model(x_noisy)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_clean(model, loader, device, snr_db=10.0, progress=True):
    """
    Evaluate clean accuracy at a fixed SNR.

    Filters samples by native dataset SNR label. No additional AWGN is applied
    (RadioML samples already have noise at their labeled SNR).
    """
    model.eval()
    correct, total = 0, 0
    for batch in iter_batches(loader, f"defense-eval clean SNR{snr_db:.0f}", progress):
        x = batch["iq"].to(device)
        y = batch["label"].to(device)
        snr_vals = batch["snr"]
        # Filter to target SNR bin
        mask = (snr_vals >= snr_db - 1.0) & (snr_vals <= snr_db + 1.0)
        if mask.sum() == 0:
            continue
        x, y = x[mask], y[mask]
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_robust(model, loader, device, rho=0.01, pgd_steps=10, snr_db=10.0,
                    progress=True):
    """
    Evaluate robust accuracy (PGD attack) at a fixed SNR.

    Filters samples by native dataset SNR label. No additional AWGN is applied.
    """
    model.eval()
    correct_clean, correct_adv, total, fooled = 0, 0, 0, 0

    for batch in iter_batches(loader, f"defense-eval PGD SNR{snr_db:.0f}", progress):
        x = batch["iq"].to(device)
        y = batch["label"].to(device)
        snr_vals = batch["snr"]
        # Filter to target SNR bin
        mask = (snr_vals >= snr_db - 1.0) & (snr_vals <= snr_db + 1.0)
        if mask.sum() == 0:
            continue
        x, y = x[mask], y[mask]

        # Clean prediction -- native samples at this SNR
        with torch.no_grad():
            logits_clean = model(x)
        preds_clean = logits_clean.argmax(1)
        correct_clean += (preds_clean == y).sum().item()

        # Adversarial attack -- no channel (AWGN already in data)
        x_adv = pgd_attack(model, x, y, rho=rho, num_steps=pgd_steps)
        with torch.no_grad():
            logits_adv = model(x_adv)
        preds_adv = logits_adv.argmax(1)
        correct_adv += (preds_adv == y).sum().item()

        correct_mask = (preds_clean == y)
        fooled += ((preds_adv != y) & correct_mask).sum().item()
        total += y.size(0)

    if total == 0:
        return {"clean_acc": 0.0, "robust_acc": 0.0, "asr": 0.0}

    return {
        "clean_acc": correct_clean / total,
        "robust_acc": correct_adv / total,
        "asr": fooled / max(correct_clean, 1),
    }


# ============================================================
# Full training pipeline for a single defense
# ============================================================

def train_model(defense_type, train_loader, val_loader, device, save_dir,
                epochs=60, lr=0.001, weight_decay=1e-4, seed=42, snr_range=(-10, 18),
                num_classes=11, input_length=128, **kwargs):
    """Train a model with the specified defense."""
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    model = RFClassifierCNN(num_classes=num_classes, input_length=input_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Build channel for augmentation-based defenses
    channel = None
    if defense_type in ["channel_aug", "adv_train_channel", "noise_inject_channel"]:
        # NOTE: RadioML samples already contain AWGN at their labeled SNR.
        # Only apply fading and CFO to avoid double-noising.
        channel = CompositeChannel([
            RayleighFadingChannel(),
            CFOChannel(max_offset=0.01),
        ]).to(device)

    best_val_acc = 0.0
    history = []
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Training: {defense_type} ({epochs} epochs)")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        if defense_type in ["baseline", "channel_aug"]:
            train_loss, train_acc = train_epoch_clean(
                model, train_loader, optimizer, criterion, device,
                channel=channel, snr_range=snr_range if channel else None)
            log = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}

        elif defense_type in ["adv_train", "adv_train_channel"]:
            rho = kwargs.get("rho", 0.01)
            pgd_steps = kwargs.get("pgd_steps", 5)
            alpha = kwargs.get("alpha", 0.5)
            train_loss, train_acc_clean, train_acc_adv = train_epoch_adversarial(
                model, train_loader, optimizer, criterion, device,
                rho=rho, pgd_steps=pgd_steps, alpha=alpha,
                channel=channel, snr_range=snr_range if channel else None)
            train_acc = train_acc_clean
            log = {"epoch": epoch, "train_loss": train_loss,
                   "train_acc_clean": train_acc_clean, "train_acc_adv": train_acc_adv}

        elif defense_type in ["noise_inject", "noise_inject_channel"]:
            sigma = kwargs.get("sigma", 0.1)
            train_loss, train_acc = train_epoch_noise_injection(
                model, train_loader, optimizer, criterion, device,
                sigma=sigma, channel=channel,
                snr_range=snr_range if channel else None)
            log = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}

        # Validate
        val_clean = evaluate_clean(model, val_loader, device, snr_db=10.0)

        # For adversarial training, use robust accuracy for model selection
        if defense_type in ["adv_train", "adv_train_channel"]:
            val_robust = evaluate_robust(model, val_loader, device,
                                         rho=kwargs.get("rho", 0.01),
                                         pgd_steps=3, snr_db=10.0)
            # Weighted: 40% clean + 60% robust (prioritize robustness)
            val_acc = 0.4 * val_clean + 0.6 * val_robust["robust_acc"]
        else:
            val_acc = val_clean

        log["val_acc"] = val_acc
        log["val_clean"] = val_clean
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_clean": val_clean,
                "defense_type": defense_type,
            }, os.path.join(save_dir, "best_model.pth"))

        history.append(log)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Best: {best_val_acc:.4f} | Time: {elapsed:.0f}s")

    total_time = time.time() - start_time
    print(f"  Done in {total_time:.1f}s ({total_time/3600:.3f} hours). Best val: {best_val_acc:.4f}")

    # Save history
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    # Save final model
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "defense_type": defense_type,
    }, os.path.join(save_dir, "final_model.pth"))

    metadata = {
        "defense_type": defense_type,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "num_classes": num_classes,
        "input_length": input_length,
        "best_val_acc": best_val_acc,
        "train_time_s": total_time,
        "kwargs": kwargs,
    }
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return model, total_time, best_val_acc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Defense Models (Phase 3)")
    parser.add_argument("--data_path", type=str, default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--dataset_version", type=str, default="2016.10a", choices=["2016.10a", "2018.01a"])
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--input_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rho", type=float, default=0.01, help="PGD budget for adv training")
    parser.add_argument("--pgd_steps", type=int, default=5, help="PGD inner steps for adv training")
    parser.add_argument("--sigma", type=float, default=0.01, help="Noise std for noise injection")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    parser.add_argument("--eval_snr", type=float, nargs="+", default=[0, 10],
                        help="SNR values for defense evaluation")
    parser.add_argument("--defenses", type=str, nargs="+", default=None,
                        help="Only train these defenses (e.g. --defenses noise_inject)")
    parser.add_argument("--skip_defense_eval", action="store_true",
                        help="Skip post-training defense comparison (useful during tuning)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm during defense evaluation (clean + PGD)")
    parser.add_argument("--progress-file", type=str, default=None,
                        help="JSON status during defense eval (default: <output_dir>/defense_eval_progress.json)")
    args = parser.parse_args()

    show_progress = not args.no_progress
    progress_file = None
    if show_progress:
        progress_file = args.progress_file or os.path.join(
            args.output_dir, "defense_eval_progress.json")

    device = get_device(args.device)
    print(f"Device: {device}")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=args.data_path, dataset_version=args.dataset_version,
        snr_range=None, batch_size=args.batch_size,  # Load all SNRs; eval functions filter per-SNR
        num_workers=0, seed=args.seed,
    )
    print(f"Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | Test: {len(test_loader.dataset):,}")

    # Define defense configs
    defenses = {
        "adv_train": {
            "type": "adv_train",
            "kwargs": {"rho": args.rho, "pgd_steps": args.pgd_steps, "alpha": 0.5},
        },
        "noise_inject": {
            "type": "noise_inject",
            "kwargs": {"sigma": args.sigma},
        },
        "channel_aug": {
            "type": "channel_aug",
            "kwargs": {},
        },
        "adv_train_channel": {
            "type": "adv_train_channel",
            "kwargs": {"rho": args.rho, "pgd_steps": args.pgd_steps, "alpha": 0.5},
        },
        "noise_inject_channel": {
            "type": "noise_inject_channel",
            "kwargs": {"sigma": args.sigma},
        },
    }

    # Filter defenses if specified
    if args.defenses:
        defenses = {k: v for k, v in defenses.items() if k in args.defenses}
        print(f"Running selected defenses: {list(defenses.keys())}")

    # Train all defenses
    trained = {}
    for name, cfg in defenses.items():
        save_dir = os.path.join(args.output_dir, name)
        model, train_time, best_val = train_model(
            defense_type=cfg["type"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_dir=save_dir,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            num_classes=args.num_classes,
            input_length=args.input_length,
            **cfg["kwargs"],
        )
        trained[name] = {
            "model": model,
            "train_time": train_time,
            "best_val": best_val,
            "params": model.get_param_count(),
        }

    if args.skip_defense_eval:
        print("\nSkipping defense comparison after training.")
        summary_rows = [
            {
                "defense": name,
                "best_val_acc": info["best_val"],
                "train_time_s": info["train_time"],
                "params": info["params"],
            }
            for name, info in trained.items()
        ]
        summary_df = pd.DataFrame(summary_rows).sort_values("best_val_acc", ascending=False)
        summary_path = os.path.join(args.output_dir, "defense_tuning_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Defense tuning summary saved to: {summary_path}")
        return

    # Load ALL existing defense checkpoints for evaluation (not just newly trained ones)
    print(f"\n{'='*60}")
    print("DEFENSE EVALUATION")
    print(f"{'='*60}")

    all_defense_names = ["adv_train", "noise_inject", "channel_aug", "adv_train_channel", "noise_inject_channel"]
    for dname in all_defense_names:
        if dname not in trained:
            ckpt_path = os.path.join(args.output_dir, dname, "best_model.pth")
            if os.path.exists(ckpt_path):
                print(f"  Loading existing checkpoint: {dname}")
                m = RFClassifierCNN(num_classes=args.num_classes, input_length=args.input_length).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                m.load_state_dict(ckpt["model_state_dict"])
                hist_path = os.path.join(args.output_dir, dname, "training_history.csv")
                t_time = 0
                if os.path.exists(hist_path):
                    # Try to recover train time from previous run
                    pass
                trained[dname] = {
                    "model": m,
                    "train_time": t_time,
                    "best_val": ckpt.get("val_acc", 0),
                    "params": m.get_param_count(),
                }

    # Try to load baseline
    baseline_path = os.path.join(args.output_dir, "baseline_cnn", "best_model.pth")
    if os.path.exists(baseline_path):
        baseline_model = RFClassifierCNN(num_classes=args.num_classes, input_length=args.input_length).to(device)
        ckpt = torch.load(baseline_path, map_location=device, weights_only=False)
        baseline_model.load_state_dict(ckpt["model_state_dict"])
        trained["baseline"] = {
            "model": baseline_model,
            "train_time": 0,
            "best_val": ckpt.get("val_acc", 0),
            "params": baseline_model.get_param_count(),
        }

    comparison_rows = []
    if progress_file:
        print(f"Defense eval progress file: {progress_file}")
    write_status(progress_file, mode="defense_eval_started", eval_snrs=[float(s) for s in args.eval_snr])

    for snr_i, snr in enumerate(args.eval_snr):
        print(f"\n  SNR = {snr} dB:")
        for name, info in trained.items():
            model = info["model"]
            model.eval()

            # Clean accuracy (filtered by native SNR, no double-noising)
            clean_acc = evaluate_clean(model, test_loader, device, snr_db=snr,
                                       progress=show_progress)

            # Robust accuracy (PGD-10, rho=1%, filtered by native SNR)
            robust = evaluate_robust(model, test_loader, device,
                                     rho=args.rho, pgd_steps=10, snr_db=snr,
                                     progress=show_progress)

            row = {
                "defense": name,
                "snr": snr,
                "clean_acc": clean_acc,
                "robust_acc": robust["robust_acc"],
                "attack_success_rate": robust["asr"],
                "train_time_s": info["train_time"],
                "params": info["params"],
            }
            comparison_rows.append(row)

            print(f"    {name:25s} | Clean: {clean_acc:.4f} | "
                  f"Robust: {robust['robust_acc']:.4f} | ASR: {robust['asr']:.4f}")
            write_status(
                progress_file,
                mode="defense_eval_pair",
                snr_db=float(snr),
                snr_index=snr_i + 1,
                total_snrs=len(args.eval_snr),
                defense=name,
                clean_acc=float(clean_acc),
                robust_acc=float(robust["robust_acc"]),
                asr=float(robust["asr"]),
            )

    # Save comparison
    df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(args.output_dir, "defense_comparison.csv")
    df.to_csv(comp_path, index=False)
    write_status(progress_file, mode="defense_eval_complete", rows=len(comparison_rows))
    print(f"\nDefense comparison saved to: {comp_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("DEFENSE SUMMARY")
    print(f"{'='*60}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
