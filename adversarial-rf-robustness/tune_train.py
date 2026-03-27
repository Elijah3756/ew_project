#!/usr/bin/env python3
"""
Grid search helper for baseline train.py runs.

This search ranks configurations by validation accuracy and emits the staged
artifacts that run_tuning_workflow.py expects for its coarse-to-fine search.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from typing import Any

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(SCRIPT_DIR, "train.py")


def _safe_value(value: float | int) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in record.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def build_trial_matrix(
    *,
    lrs: list[float],
    batch_sizes: list[int],
    epochs_list: list[int],
    weight_decays: list[float],
) -> list[dict[str, Any]]:
    return [
        {
            "lr": float(lr),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "weight_decay": float(weight_decay),
        }
        for lr, batch_size, epochs, weight_decay in itertools.product(
            lrs,
            batch_sizes,
            epochs_list,
            weight_decays,
        )
    ]


def rank_top_results(df: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    ok = df[df["status"] == "ok"].copy()
    if len(ok) == 0:
        return []
    ok = ok.sort_values(
        ["best_val_acc", "epochs", "batch_size", "lr"],
        ascending=[False, True, True, True],
    )
    return ok.head(top_k).to_dict(orient="records")


def _sorted_unique_float(values: list[float]) -> list[float]:
    return sorted({float(v) for v in values})


def _sorted_unique_int(values: list[int]) -> list[int]:
    return sorted({int(v) for v in values if int(v) > 0})


def build_refined_candidates(
    top_rows: list[dict[str, Any]],
    *,
    lr_factors: list[float],
    batch_multipliers: list[float],
    epoch_offsets: list[int],
    weight_decay_factors: list[float],
) -> dict[str, list[Any]]:
    lrs: list[float] = []
    batch_sizes: list[int] = []
    epochs: list[int] = []
    weight_decays: list[float] = []

    for row in top_rows:
        base_lr = float(row["lr"])
        base_batch = int(row["batch_size"])
        base_epochs = int(row["epochs"])
        base_weight_decay = float(row["weight_decay"])

        lrs.extend(base_lr * factor for factor in lr_factors if base_lr * factor > 0)
        batch_sizes.extend(max(1, int(round(base_batch * factor))) for factor in batch_multipliers)
        epochs.extend(max(1, base_epochs + offset) for offset in epoch_offsets)

        if base_weight_decay == 0.0:
            weight_decays.append(0.0)
        else:
            weight_decays.extend(
                base_weight_decay * factor
                for factor in weight_decay_factors
                if base_weight_decay * factor >= 0
            )

    return {
        "lr": _sorted_unique_float(lrs),
        "batch_size": _sorted_unique_int(batch_sizes),
        "epochs": _sorted_unique_int(epochs),
        "weight_decay": _sorted_unique_float(weight_decays),
    }


def _tag_for_config(config: dict[str, Any]) -> str:
    return "_".join(
        [
            f"lr{_safe_value(config['lr'])}",
            f"bs{_safe_value(config['batch_size'])}",
            f"ep{_safe_value(config['epochs'])}",
            f"wd{_safe_value(config['weight_decay'])}",
        ]
    )


def _run_one_trial(config: dict[str, Any], args: argparse.Namespace, search_root: str) -> dict[str, Any]:
    tag = _tag_for_config(config)
    trial_root = os.path.join(search_root, tag)
    os.makedirs(trial_root, exist_ok=True)
    cmd = [
        sys.executable,
        TRAIN_PY,
        "--data_path",
        args.data_path,
        "--dataset_version",
        args.dataset_version,
        "--epochs",
        str(config["epochs"]),
        "--batch_size",
        str(config["batch_size"]),
        "--lr",
        str(config["lr"]),
        "--weight_decay",
        str(config["weight_decay"]),
        "--seed",
        str(args.seed),
        "--save_dir",
        trial_root,
        "--num_classes",
        str(args.num_classes),
        "--input_length",
        str(args.input_length),
        "--device",
        args.device,
        "--subset_fraction",
        str(args.subset_fraction),
        "--skip_snr_sweep",
    ]
    if args.channel_aug:
        cmd.append("--channel_aug")

    start = time.time()
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    elapsed = time.time() - start

    row = {
        **config,
        "trial_tag": tag,
        "trial_root": trial_root,
        "best_model_path": os.path.join(trial_root, "best_model.pth"),
    }
    if result.returncode != 0:
        row.update({"best_val_acc": None, "final_val_acc": None, "train_time_s": elapsed, "status": f"failed_exit_{result.returncode}"})
        return row

    history_path = os.path.join(trial_root, "training_history.csv")
    if not os.path.isfile(history_path):
        row.update({"best_val_acc": None, "final_val_acc": None, "train_time_s": elapsed, "status": "missing_history"})
        return row

    history = pd.read_csv(history_path)
    row.update(
        {
            "best_val_acc": float(history["val_acc"].max()),
            "final_val_acc": float(history["val_acc"].iloc[-1]),
            "train_time_s": elapsed,
            "status": "ok",
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search over baseline training settings")
    parser.add_argument("--data_path", type=str, default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--dataset_version", type=str, default="2016.10a")
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--input_length", type=int, default=128)
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-3, 5e-4])
    parser.add_argument("--batch_size", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--epochs", type=int, nargs="+", default=[60, 100])
    parser.add_argument("--weight_decay", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--output_root", type=str, default="experiments/tune_train")
    parser.add_argument("--search_name", type=str, default="grid")
    parser.add_argument("--top_k_summary", type=int, default=10)
    parser.add_argument("--refine_top_k", type=int, default=5)
    parser.add_argument("--refine_lr_factors", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--refine_batch_multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--refine_epoch_offsets", type=int, nargs="+", default=[-20, 0, 20])
    parser.add_argument("--refine_weight_decay_factors", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--channel_aug", action="store_true")
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=1.0,
        help="Fraction of training data per trial (stratified). "
             "E.g. 0.10 uses 10%% of train set for faster HP search.",
    )
    parser.add_argument("--emit_refine_candidates", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    search_root = os.path.join(args.output_root, args.search_name)
    os.makedirs(search_root, exist_ok=True)

    rows: list[dict[str, Any]] = []
    trials = build_trial_matrix(
        lrs=args.lr,
        batch_sizes=args.batch_size,
        epochs_list=args.epochs,
        weight_decays=args.weight_decay,
    )
    print(f"=== {args.search_name} ({len(trials)} runs) ===")
    for i, config in enumerate(trials):
        print(f"[{i + 1}/{len(trials)}] {_tag_for_config(config)}")
        rows.append(_run_one_trial(config, args, search_root))

    df = pd.DataFrame(rows)
    results_csv_path = os.path.join(args.output_root, f"{args.search_name}_results.csv")
    df.to_csv(results_csv_path, index=False)

    ranked = rank_top_results(df, top_k=args.top_k_summary)
    if not ranked:
        print(f"\nNo successful baseline tuning runs. See {results_csv_path}")
        return

    top_path = os.path.join(args.output_root, f"top_{args.search_name}.json")
    with open(top_path, "w") as f:
        json.dump([_clean_record(row) for row in ranked], f, indent=2)

    best_path = os.path.join(args.output_root, f"best_{args.search_name}.json")
    with open(best_path, "w") as f:
        json.dump(_clean_record(ranked[0]), f, indent=2)

    if args.emit_refine_candidates:
        refine_path = os.path.join(args.output_root, f"refine_candidates_{args.search_name}.json")
        candidates = build_refined_candidates(
            ranked[: args.refine_top_k],
            lr_factors=args.refine_lr_factors,
            batch_multipliers=args.refine_batch_multipliers,
            epoch_offsets=args.refine_epoch_offsets,
            weight_decay_factors=args.refine_weight_decay_factors,
        )
        with open(refine_path, "w") as f:
            json.dump(candidates, f, indent=2)
        print(f"Refine candidates: {refine_path}")

    print("\n--- Top baseline configurations ---")
    print(pd.DataFrame(ranked).to_string(index=False))
    print(f"\nFull table: {results_csv_path}")
    print(f"Best config: {best_path}")
    print(f"Top configs: {top_path}")


if __name__ == "__main__":
    main()
