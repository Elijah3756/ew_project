#!/usr/bin/env python3
"""
Sweep defense-specific hyperparameters for train_defenses.py.

Tunes:
  - adv_train / adv_train_channel: rho, pgd_steps
  - noise_inject / noise_inject_channel: sigma

Each run trains a SINGLE defense and records defense_comparison.csv.
Selection metric: robust_acc at the highest eval SNR.

Usage:
  python tune_defenses.py \
    --data_path data/raw/RML2016.10a_dict.pkl \
    --rho 0.005 0.01 0.02 \
    --pgd_steps 3 5 7 \
    --sigma 0.005 0.01 0.02 \
    --output_root experiments/tune_defenses
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DEFENSES = os.path.join(SCRIPT_DIR, "train_defenses.py")


def _tag(*parts: str) -> str:
    return "_".join(str(p) for p in parts).replace(".", "p").replace("-", "m")


def _run_defense(
    defense: str,
    data_path: str,
    dataset_version: str,
    num_classes: int,
    input_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    output_dir: str,
    eval_snr: list[float],
    rho: float = 0.01,
    pgd_steps: int = 5,
    sigma: float = 0.01,
) -> int:
    cmd = [
        sys.executable,
        TRAIN_DEFENSES,
        "--data_path", data_path,
        "--dataset_version", dataset_version,
        "--num_classes", str(num_classes),
        "--input_length", str(input_length),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--seed", str(seed),
        "--device", device,
        "--rho", str(rho),
        "--pgd_steps", str(pgd_steps),
        "--sigma", str(sigma),
        "--output_dir", output_dir,
        "--eval_snr", *[str(s) for s in eval_snr],
        "--defenses", defense,
    ]
    r = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return r.returncode


def _read_best_robust(output_dir: str, defense: str, eval_snr: list[float]) -> dict | None:
    csv_path = os.path.join(output_dir, "defense_comparison.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df = df[df["defense"] == defense]
    if df.empty:
        return None
    max_snr = max(eval_snr)
    row = df[df["snr"] == max_snr]
    if row.empty:
        row = df.iloc[-1:]
    row = row.iloc[0]
    return {
        "clean_acc": float(row["clean_acc"]),
        "robust_acc": float(row["robust_acc"]),
        "asr": float(row["attack_success_rate"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep defense hyperparameters")
    p.add_argument("--data_path", type=str, default="data/raw/RML2016.10a_dict.pkl")
    p.add_argument("--dataset_version", type=str, default="2016.10a")
    p.add_argument("--num_classes", type=int, default=11)
    p.add_argument("--input_length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval_snr", type=float, nargs="+", default=[0, 10])
    p.add_argument("--output_root", type=str, default="experiments/tune_defenses")
    p.add_argument(
        "--rho", type=float, nargs="+", default=[0.005, 0.01, 0.02],
        help="PGD rho values to sweep for adv_train",
    )
    p.add_argument(
        "--pgd_steps", type=int, nargs="+", default=[3, 5, 7],
        help="PGD inner steps to sweep for adv_train",
    )
    p.add_argument(
        "--sigma", type=float, nargs="+", default=[0.005, 0.01, 0.02],
        help="Noise std values to sweep for noise_inject",
    )
    args = p.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    rows: list[dict] = []

    # --- Adversarial training sweep (rho x pgd_steps) ---
    adv_combos = list(itertools.product(args.rho, args.pgd_steps))
    print(f"Adv training sweep: {len(adv_combos)} combos (rho x pgd_steps)")

    for i, (rho, steps) in enumerate(adv_combos):
        tag = _tag("adv", f"rho{rho}", f"steps{steps}")
        out = os.path.join(args.output_root, tag)

        print(f"\n[adv {i + 1}/{len(adv_combos)}] rho={rho} pgd_steps={steps} -> {out}")
        rc = _run_defense(
            defense="adv_train",
            data_path=args.data_path,
            dataset_version=args.dataset_version,
            num_classes=args.num_classes,
            input_length=args.input_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            output_dir=out,
            eval_snr=args.eval_snr,
            rho=rho,
            pgd_steps=steps,
        )
        metrics = _read_best_robust(out, "adv_train", args.eval_snr) if rc == 0 else None
        rows.append({
            "defense": "adv_train",
            "rho": rho,
            "pgd_steps": steps,
            "sigma": None,
            "save_dir": out,
            **(metrics or {"clean_acc": None, "robust_acc": None, "asr": None}),
            "status": "ok" if rc == 0 and metrics else f"fail_{rc}",
        })

    # --- Noise injection sweep (sigma) ---
    print(f"\nNoise injection sweep: {len(args.sigma)} sigma values")

    for i, sigma in enumerate(args.sigma):
        tag = _tag("noise", f"sigma{sigma}")
        out = os.path.join(args.output_root, tag)

        print(f"\n[noise {i + 1}/{len(args.sigma)}] sigma={sigma} -> {out}")
        rc = _run_defense(
            defense="noise_inject",
            data_path=args.data_path,
            dataset_version=args.dataset_version,
            num_classes=args.num_classes,
            input_length=args.input_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            output_dir=out,
            eval_snr=args.eval_snr,
            sigma=sigma,
        )
        metrics = _read_best_robust(out, "noise_inject", args.eval_snr) if rc == 0 else None
        rows.append({
            "defense": "noise_inject",
            "rho": None,
            "pgd_steps": None,
            "sigma": sigma,
            "save_dir": out,
            **(metrics or {"clean_acc": None, "robust_acc": None, "asr": None}),
            "status": "ok" if rc == 0 and metrics else f"fail_{rc}",
        })

    # --- Save and display ---
    df = pd.DataFrame(rows)
    csv_out = os.path.join(args.output_root, "tune_defense_results.csv")
    df.to_csv(csv_out, index=False)

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("\nNo successful runs. Check tune_defense_results.csv")
        return

    print("\n" + "=" * 70)
    print("DEFENSE TUNING RESULTS (sorted by robust_acc)")
    print("=" * 70)

    for defense in ["adv_train", "noise_inject"]:
        sub = ok[ok["defense"] == defense].sort_values("robust_acc", ascending=False)
        if sub.empty:
            continue
        print(f"\n--- {defense} ---")
        cols = ["rho", "pgd_steps", "sigma", "clean_acc", "robust_acc", "asr"]
        print(sub[cols].to_string(index=False))
        best = sub.iloc[0]
        print(f"  Best: robust_acc={best['robust_acc']:.4f}")

    print(f"\nFull table: {csv_out}")


if __name__ == "__main__":
    main()
