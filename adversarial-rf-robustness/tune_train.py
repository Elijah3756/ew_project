#!/usr/bin/env python3
"""
Grid search over learning rate, batch size, and epochs for train.py.

Each run uses the same train/val split; selection metric is the maximum
validation accuracy recorded in training_history.csv (same criterion as
saving best_model.pth).

Skips the expensive test-set SNR sweep (--skip_snr_sweep) per trial.

Usage:
  python tune_train.py \\
    --data_path data/raw/RML2016.10a_dict.pkl \\
    --lr 0.001 0.0005 0.0001 \\
    --batch_size 128 256 \\
    --epochs 40 60 \\
    --output_root experiments/tune_baseline
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(SCRIPT_DIR, "train.py")


def _safe_tag(lr: float, bs: int, ep: int) -> str:
    s = f"lr{lr:g}_bs{bs}_ep{ep}"
    return s.replace(".", "p").replace("-", "m")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Grid search lr, batch_size, epochs (train.py, val accuracy)"
    )
    p.add_argument("--data_path", type=str, default="data/raw/RML2016.10a_dict.pkl")
    p.add_argument("--dataset_version", type=str, default="2016.10a")
    p.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=[1e-3, 5e-4, 1e-4],
        help="Learning rates to try",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[128, 256],
        help="Batch sizes to try",
    )
    p.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[40, 60, 100],
        help="Epoch counts to try",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="experiments/tune_baseline",
        help="Directory containing one subfolder per configuration",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_classes", type=int, default=11)
    p.add_argument("--input_length", type=int, default=128)
    p.add_argument("--snr_min", type=int, default=-10)
    p.add_argument("--snr_max", type=int, default=18)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--channel_aug", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    combos = list(itertools.product(args.lr, args.batch_size, args.epochs))
    n = len(combos)
    print(f"Grid: {len(args.lr)} x {len(args.batch_size)} x {len(args.epochs)} = {n} runs")
    print("(SNR sweep disabled per run for speed.)\n")

    rows: list[dict] = []

    for i, (lr, bs, ep) in enumerate(combos):
        tag = _safe_tag(lr, bs, ep)
        save_dir = os.path.join(args.output_root, tag)

        cmd = [
            sys.executable,
            TRAIN_PY,
            "--data_path",
            args.data_path,
            "--dataset_version",
            args.dataset_version,
            "--lr",
            str(lr),
            "--batch_size",
            str(bs),
            "--epochs",
            str(ep),
            "--save_dir",
            save_dir,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--num_classes",
            str(args.num_classes),
            "--input_length",
            str(args.input_length),
            "--snr_min",
            str(args.snr_min),
            "--snr_max",
            str(args.snr_max),
            "--weight_decay",
            str(args.weight_decay),
            "--skip_snr_sweep",
        ]
        if args.channel_aug:
            cmd.append("--channel_aug")

        print(f"[{i + 1}/{n}] lr={lr} batch_size={bs} epochs={ep} -> {save_dir}")

        r = subprocess.run(cmd, cwd=SCRIPT_DIR)
        if r.returncode != 0:
            rows.append(
                {
                    "lr": lr,
                    "batch_size": bs,
                    "epochs": ep,
                    "save_dir": save_dir,
                    "best_val_acc": None,
                    "final_val_acc": None,
                    "status": f"failed_exit_{r.returncode}",
                }
            )
            continue

        hist_path = os.path.join(save_dir, "training_history.csv")
        if not os.path.isfile(hist_path):
            rows.append(
                {
                    "lr": lr,
                    "batch_size": bs,
                    "epochs": ep,
                    "save_dir": save_dir,
                    "best_val_acc": None,
                    "final_val_acc": None,
                    "status": "missing_history",
                }
            )
            continue

        hist = pd.read_csv(hist_path)
        best_val = float(hist["val_acc"].max())
        final_val = float(hist["val_acc"].iloc[-1])
        rows.append(
            {
                "lr": lr,
                "batch_size": bs,
                "epochs": ep,
                "save_dir": save_dir,
                "best_val_acc": best_val,
                "final_val_acc": final_val,
                "status": "ok",
            }
        )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_root, "tune_results.csv")
    df.to_csv(out_csv, index=False)

    ok = df[df["status"] == "ok"].copy()
    if len(ok) == 0:
        print("\nNo successful runs. See tune_results.csv")
        return

    ok = ok.sort_values("best_val_acc", ascending=False)
    print("\n--- Top configurations (by max validation accuracy) ---")
    print(ok.head(15).to_string(index=False))
    best = ok.iloc[0]
    print(
        f"\nBest: lr={best['lr']}, batch_size={int(best['batch_size'])}, "
        f"epochs={int(best['epochs'])}, best_val_acc={best['best_val_acc']:.4f}"
    )
    print(f"\nFull table: {out_csv}")


if __name__ == "__main__":
    main()
