"""
Training script for RF modulation classifier.
Supports clean training, channel-augmented training, and adversarial training.
"""

import os
import sys
import argparse
import time
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import get_dataloaders
from models.cnn import RFClassifierCNN
from channels.awgn import AWGNChannel
from channels.rayleigh import RayleighFadingChannel
from channels.cfo import CFOChannel
from channels.composite import CompositeChannel


def get_device(device_str: str = "auto") -> torch.device:
    """Detect best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_channel_augmentation(config: dict):
    """Build channel augmentation from config."""
    if not config.get("channel_augmentation", {}).get("enabled", False):
        return None

    layers = []
    ch_cfg = config["channel_augmentation"]

    if ch_cfg.get("rayleigh", False):
        layers.append(RayleighFadingChannel(
            coherence_samples=ch_cfg.get("rayleigh_coherence", None)
        ))

    if ch_cfg.get("cfo", False):
        layers.append(CFOChannel(
            max_offset=ch_cfg.get("cfo_max_offset", 0.01)
        ))

    if ch_cfg.get("awgn", False):
        # WARNING: RadioML samples already contain AWGN at their labeled SNR.
        # Adding AWGNChannel here causes double-noising. Only enable this for
        # synthetic datasets where samples are noiseless.
        import warnings
        warnings.warn(
            "AWGNChannel enabled in channel augmentation. If using RadioML data "
            "(which has baked-in AWGN), this causes double-noising. "
            "Set awgn: false in config for RadioML datasets.",
            UserWarning
        )
        layers.append(AWGNChannel(
            snr_db=ch_cfg.get("awgn_snr_db", 10.0)
        ))

    if layers:
        return CompositeChannel(layers)
    return None


def train_epoch(model, train_loader, optimizer, criterion, device, channel=None, snr_range=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        x = batch["iq"].to(device)
        y = batch["label"].to(device)

        # Apply channel augmentation if enabled
        if channel is not None:
            if snr_range is not None:
                # Random SNR per batch during training
                snr = np.random.uniform(snr_range[0], snr_range[1])
                x = channel(x, snr_db=snr)
            else:
                x = channel(x)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, channel=None, snr_db=None):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["iq"].to(device)
        y = batch["label"].to(device)

        if channel is not None and snr_db is not None:
            x = channel(x, snr_db=snr_db)
        elif channel is not None:
            x = channel(x)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_snr_sweep(model, loader, device, snr_values):
    """
    Evaluate model across SNR values.

    Filters samples by native dataset SNR label. No additional AWGN is applied
    (RadioML samples already have noise at their labeled SNR level).
    """
    model.eval()
    results = {}

    for snr in snr_values:
        correct = 0
        total = 0

        for batch in loader:
            x = batch["iq"].to(device)
            y = batch["label"].to(device)
            snr_vals = batch["snr"]

            # Filter to target SNR bin
            mask = (snr_vals >= snr - 1.0) & (snr_vals <= snr + 1.0)
            if mask.sum() == 0:
                continue
            x, y = x[mask], y[mask]

            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        results[snr] = correct / total if total > 0 else 0.0

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RF Modulation Classifier")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--data_path", type=str, default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--dataset_version", type=str, default="2016.10a")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="experiments/results/baseline_cnn")
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--input_length", type=int, default=128)
    parser.add_argument("--snr_min", type=int, default=-10)
    parser.add_argument("--snr_max", type=int, default=18)
    parser.add_argument("--channel_aug", action="store_true", help="Enable channel augmentation")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0-1.0). Stratified by class. "
             "Useful for faster HP tuning. Val/test remain full-sized.",
    )
    parser.add_argument(
        "--skip_snr_sweep",
        action="store_true",
        help="Skip post-training clean SNR sweep (useful for tuning runs).",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Set seed
    seed = config.get("experiment", {}).get("seed", args.seed)
    set_seed(seed)

    # Device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Data
    data_path = config.get("dataset", {}).get("path", args.data_path)
    dataset_version = config.get("dataset", {}).get("version", args.dataset_version)
    snr_range = config.get("dataset", {}).get("snr_range", [args.snr_min, args.snr_max])
    if isinstance(snr_range, list):
        snr_range = tuple(snr_range)
    batch_size = config.get("training", {}).get("batch_size", args.batch_size)
    num_workers = config.get("training", {}).get("num_workers", 0)

    subset_fraction = config.get("training", {}).get("subset_fraction", args.subset_fraction)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=data_path,
        dataset_version=dataset_version,
        snr_range=snr_range,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        subset_fraction=subset_fraction,
    )
    print(f"Dataset: {dataset_version}, SNR range: {snr_range}")
    if subset_fraction < 1.0:
        print(f"*** HP-tuning mode: using {subset_fraction:.0%} stratified subset of training data ***")
    print(f"Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | Test: {len(test_loader.dataset):,}")

    # Model
    num_classes = config.get("model", {}).get("num_classes", args.num_classes)
    input_length = config.get("model", {}).get("input_length", args.input_length)
    dropout = config.get("model", {}).get("dropout", 0.5)
    model = RFClassifierCNN(num_classes=num_classes, input_length=input_length, dropout=dropout)
    model = model.to(device)
    print(f"Model parameters: {model.get_param_count():,}")

    # Channel augmentation
    channel = None
    if args.channel_aug:
        # NOTE: RadioML samples already contain AWGN at their labeled SNR.
        # We only apply fading and CFO augmentation to avoid double-noising.
        # The native dataset noise serves as the AWGN component.
        channel = CompositeChannel([
            RayleighFadingChannel(),
            CFOChannel(max_offset=0.01),
        ]).to(device)
        print(f"Channel augmentation: {channel}")
    else:
        channel_from_config = build_channel_augmentation(config)
        if channel_from_config:
            channel = channel_from_config.to(device)
            print(f"Channel augmentation (from config): {channel}")

    # Training setup
    epochs = config.get("training", {}).get("epochs", args.epochs)
    lr = config.get("training", {}).get("learning_rate", args.lr)
    wd = config.get("training", {}).get("weight_decay", args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Save directory
    save_dir = config.get("logging", {}).get("save_dir", args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 70)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            channel=channel,
            snr_range=snr_range if channel is not None else None,
        )

        # Validate (clean, no channel)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, os.path.join(save_dir, "best_model.pth"))

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Best: {best_val_acc:.4f} | "
                  f"Time: {elapsed:.0f}s")

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"Training complete in {total_time:.1f}s ({total_time/3600:.2f} GPU hours)")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Save final model
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(save_dir, "final_model.pth"))

    # Save history
    import pandas as pd
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    if args.skip_snr_sweep:
        print("\nSkipping clean SNR sweep (--skip_snr_sweep).")
    else:
        # SNR sweep evaluation on test set (filtered by native SNR labels)
        # Reload test set with ALL SNRs for full sweep
        print("\nRunning SNR sweep evaluation on test set...")
        from data.dataset import RadioMLDataset
        from torch.utils.data import DataLoader
        test_ds_full = RadioMLDataset(
            data_path=data_path, dataset_version=dataset_version,
            snr_range=None, split="test", seed=seed)
        test_loader_full = DataLoader(test_ds_full, batch_size=batch_size, shuffle=False,
                                       num_workers=0)
        if dataset_version == "2018.01a":
            snr_values = list(range(-20, 32, 2))
        else:
            snr_values = list(range(-20, 20, 2))

        snr_results_clean = evaluate_snr_sweep(model, test_loader_full, device, snr_values)

        # Save SNR results
        snr_df = pd.DataFrame([
            {"snr": snr, "accuracy": acc}
            for snr, acc in snr_results_clean.items()
        ])
        snr_df.to_csv(os.path.join(save_dir, "snr_sweep_clean.csv"), index=False)

        print("\nClean Accuracy vs SNR:")
        for snr, acc in sorted(snr_results_clean.items()):
            bar = "#" * int(acc * 50)
            print(f"  SNR {snr:+3d} dB: {acc:.4f}  {bar}")

    # Save config used
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nResults saved to: {save_dir}/")


if __name__ == "__main__":
    main()
