#!/usr/bin/env python3
"""
Grid search helper for train_defenses.py.

Runs one defense at a time with --skip_defense_eval so tuning optimizes
validation-driven training quality rather than the expensive post-training
comparison stage.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from typing import Any

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DEFENSES_PY = os.path.join(SCRIPT_DIR, "train_defenses.py")


def _safe_value(value: float | int) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def build_defense_search_spaces(
    shared: dict[str, list[Any]],
    rho_values: list[float],
    pgd_steps_values: list[int],
    sigma_values: list[float],
) -> dict[str, list[dict[str, Any]]]:
    shared_combos = [
        {
            "lr": float(lr),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "weight_decay": float(weight_decay),
        }
        for lr, batch_size, epochs, weight_decay in itertools.product(
            shared["lr"],
            shared["batch_size"],
            shared["epochs"],
            shared["weight_decay"],
        )
    ]

    spaces: dict[str, list[dict[str, Any]]] = {
        "channel_aug": [cfg.copy() for cfg in shared_combos],
    }

    spaces["adv_train"] = [
        {**cfg, "rho": float(rho), "pgd_steps": int(pgd_steps)}
        for cfg in shared_combos
        for rho, pgd_steps in itertools.product(rho_values, pgd_steps_values)
    ]
    spaces["adv_train_channel"] = [cfg.copy() for cfg in spaces["adv_train"]]

    spaces["noise_inject"] = [
        {**cfg, "sigma": float(sigma)}
        for cfg in shared_combos
        for sigma in sigma_values
    ]
    spaces["noise_inject_channel"] = [cfg.copy() for cfg in spaces["noise_inject"]]

    return spaces


def rank_top_results(df: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    ok = df[df["status"] == "ok"].copy()
    if len(ok) == 0:
        return []
    ok = ok.sort_values(
        ["best_val_acc", "defense", "epochs"],
        ascending=[False, True, True],
    )
    return ok.head(top_k).to_dict(orient="records")


def best_results_by_defense(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    ok = df[df["status"] == "ok"].copy()
    if len(ok) == 0:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for defense in sorted(ok["defense"].unique()):
        ddf = ok[ok["defense"] == defense].copy()
        ddf = ddf.sort_values(["best_val_acc"], ascending=[False])
        result[defense] = _clean_record(ddf.iloc[0].to_dict())
    return result


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in record.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def _tag_for_config(config: dict[str, Any]) -> str:
    pieces = [
        f"lr{_safe_value(config['lr'])}",
        f"bs{_safe_value(config['batch_size'])}",
        f"ep{_safe_value(config['epochs'])}",
        f"wd{_safe_value(config['weight_decay'])}",
    ]
    if "rho" in config:
        pieces.append(f"rho{_safe_value(config['rho'])}")
    if "pgd_steps" in config:
        pieces.append(f"pgd{_safe_value(config['pgd_steps'])}")
    if "sigma" in config:
        pieces.append(f"sigma{_safe_value(config['sigma'])}")
    return "_".join(pieces)


def _run_one_trial(
    defense: str,
    config: dict[str, Any],
    args: argparse.Namespace,
    output_root: str,
) -> dict[str, Any]:
    tag = _tag_for_config(config)
    trial_root = os.path.join(output_root, defense, tag)
    cmd = [
        sys.executable,
        TRAIN_DEFENSES_PY,
        "--data_path",
        args.data_path,
        "--dataset_version",
        args.dataset_version,
        "--num_classes",
        str(args.num_classes),
        "--input_length",
        str(args.input_length),
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
        "--device",
        args.device,
        "--output_dir",
        trial_root,
        "--defenses",
        defense,
        "--skip_defense_eval",
        "--no-progress",
    ]
    if "rho" in config:
        cmd.extend(["--rho", str(config["rho"])])
    if "pgd_steps" in config:
        cmd.extend(["--pgd_steps", str(config["pgd_steps"])])
    if "sigma" in config:
        cmd.extend(["--sigma", str(config["sigma"])])

    r = subprocess.run(cmd, cwd=SCRIPT_DIR)
    result = {"defense": defense, "trial_root": trial_root, **config}
    if r.returncode != 0:
        result.update({"best_val_acc": None, "status": f"failed_exit_{r.returncode}"})
        return result

    history_path = os.path.join(trial_root, defense, "training_history.csv")
    run_config_path = os.path.join(trial_root, defense, "run_config.json")
    if not os.path.isfile(history_path):
        result.update({"best_val_acc": None, "status": "missing_history"})
        return result

    hist = pd.read_csv(history_path)
    result["best_val_acc"] = float(hist["val_acc"].max())
    result["final_val_acc"] = float(hist["val_acc"].iloc[-1])
    result["status"] = "ok"
    if os.path.isfile(run_config_path):
        with open(run_config_path, "r") as f:
            metadata = json.load(f)
        result["train_time_s"] = metadata.get("train_time_s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search over defense training settings")
    parser.add_argument("--data_path", type=str, default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--dataset_version", type=str, default="2016.10a")
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--input_length", type=int, default=128)
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-3, 5e-4])
    parser.add_argument("--batch_size", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--epochs", type=int, nargs="+", default=[60, 100])
    parser.add_argument("--weight_decay", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--rho", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--pgd_steps", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--sigma", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument(
        "--defenses",
        type=str,
        nargs="+",
        default=None,
        choices=["adv_train", "noise_inject", "channel_aug", "adv_train_channel", "noise_inject_channel"],
    )
    parser.add_argument("--output_root", type=str, default="experiments/tune_defenses")
    parser.add_argument("--search_name", type=str, default="grid")
    parser.add_argument("--top_k_summary", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    search_root = os.path.join(args.output_root, args.search_name)
    os.makedirs(search_root, exist_ok=True)

    spaces = build_defense_search_spaces(
        shared={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
        },
        rho_values=args.rho,
        pgd_steps_values=args.pgd_steps,
        sigma_values=args.sigma,
    )
    selected = args.defenses or list(spaces.keys())

    rows: list[dict[str, Any]] = []
    for defense in selected:
        configs = spaces[defense]
        print(f"\n=== {defense} ({len(configs)} runs) ===")
        for i, config in enumerate(configs):
            print(f"[{i + 1}/{len(configs)}] {defense} -> {_tag_for_config(config)}")
            rows.append(_run_one_trial(defense, config, args, search_root))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(search_root, f"{args.search_name}_results.csv")
    df.to_csv(out_csv, index=False)

    ranked = rank_top_results(df, top_k=args.top_k_summary)
    if not ranked:
        print(f"\nNo successful defense tuning runs. See {out_csv}")
        return

    print("\n--- Top defense configurations ---")
    print(pd.DataFrame(ranked).to_string(index=False))
    best_path = os.path.join(search_root, f"best_{args.search_name}.json")
    with open(best_path, "w") as f:
        json.dump(_clean_record(ranked[0]), f, indent=2)
    top_path = os.path.join(search_root, f"top_{args.search_name}.json")
    with open(top_path, "w") as f:
        json.dump([_clean_record(row) for row in ranked], f, indent=2)
    best_by_defense_path = os.path.join(search_root, f"best_by_defense_{args.search_name}.json")
    with open(best_by_defense_path, "w") as f:
        json.dump(best_results_by_defense(df), f, indent=2)
    print(f"\nFull table: {out_csv}")
    print(f"Best config: {best_path}")
    print(f"Best by defense: {best_by_defense_path}")


if __name__ == "__main__":
    main()
