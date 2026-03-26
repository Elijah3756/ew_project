#!/usr/bin/env python3
"""
Orchestrate the staged coarse-to-fine tuning workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TUNE_TRAIN_PY = os.path.join(SCRIPT_DIR, "tune_train.py")
TUNE_DEFENSES_PY = os.path.join(SCRIPT_DIR, "tune_defenses.py")


def build_stage_sequence() -> list[dict[str, Any]]:
    return [
        {"name": "baseline2016_coarse"},
        {"name": "baseline2016_refine"},
        {"name": "baseline2016_weight_decay"},
        {"name": "baseline2018_coarse"},
        {"name": "baseline2018_refine"},
        {"name": "defense_tuning"},
    ]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def _load_json(path: str, stage_name: str) -> Any:
    if not os.path.isfile(path):
        raise RuntimeError(
            f"Expected workflow artifact missing for {stage_name}: {path}. "
            "This usually means the previous stage had no successful runs."
        )
    with open(path, "r") as f:
        return json.load(f)


def _load_top_rows(output_root: str, search_name: str) -> list[dict[str, Any]]:
    top_path = os.path.join(output_root, f"top_{search_name}.json")
    rows = _load_json(top_path, search_name)
    if not rows:
        raise RuntimeError(f"No successful runs recorded for {search_name}.")
    return rows


def _load_refine_candidates(output_root: str, search_name: str) -> dict[str, list[Any]]:
    path = os.path.join(output_root, f"refine_candidates_{search_name}.json")
    candidates = _load_json(path, search_name)
    if not all(candidates.get(key) for key in ["lr", "batch_size", "epochs", "weight_decay"]):
        raise RuntimeError(f"Refine candidates for {search_name} are empty or incomplete.")
    return candidates


def _stringify(values: list[Any]) -> list[str]:
    return [str(v) for v in values]


def _run_tune_train(
    *,
    data_path: str,
    dataset_version: str,
    output_root: str,
    search_name: str,
    lr: list[float],
    batch_size: list[int],
    epochs: list[int],
    weight_decay: list[float],
    args: argparse.Namespace,
    emit_refine_candidates: bool,
) -> None:
    cmd = [
        sys.executable,
        TUNE_TRAIN_PY,
        "--data_path",
        data_path,
        "--dataset_version",
        dataset_version,
        "--output_root",
        output_root,
        "--search_name",
        search_name,
        "--lr",
        *_stringify(lr),
        "--batch_size",
        *_stringify(batch_size),
        "--epochs",
        *_stringify(epochs),
        "--weight_decay",
        *_stringify(weight_decay),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num_classes",
        str(args.num_classes_2016 if dataset_version == "2016.10a" else args.num_classes_2018),
        "--input_length",
        str(args.input_length_2016 if dataset_version == "2016.10a" else args.input_length_2018),
        "--top_k_summary",
        str(args.top_k_summary),
        "--refine_top_k",
        str(args.refine_top_k),
        "--refine_lr_factors",
        *_stringify(args.refine_lr_factors),
        "--refine_batch_multipliers",
        *_stringify(args.refine_batch_multipliers),
        "--refine_epoch_offsets",
        *_stringify(args.refine_epoch_offsets),
        "--refine_weight_decay_factors",
        *_stringify(args.refine_weight_decay_factors),
    ]
    if emit_refine_candidates:
        cmd.append("--emit_refine_candidates")
    _run(cmd)


def _rows_to_axes(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "lr": sorted({float(row["lr"]) for row in rows}),
        "batch_size": sorted({int(row["batch_size"]) for row in rows}),
        "epochs": sorted({int(row["epochs"]) for row in rows}),
        "weight_decay": sorted({float(row["weight_decay"]) for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the staged hyperparameter tuning workflow")
    parser.add_argument("--data_path_2016", type=str, required=True)
    parser.add_argument("--data_path_2018", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="experiments/tuning_workflow")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k_summary", type=int, default=10)
    parser.add_argument("--refine_top_k", type=int, default=5)
    parser.add_argument("--refine_lr_factors", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--refine_batch_multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--refine_epoch_offsets", type=int, nargs="+", default=[-20, 0, 20])
    parser.add_argument("--refine_weight_decay_factors", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--num_classes_2016", type=int, default=11)
    parser.add_argument("--input_length_2016", type=int, default=128)
    parser.add_argument("--num_classes_2018", type=int, default=24)
    parser.add_argument("--input_length_2018", type=int, default=1024)
    parser.add_argument("--coarse_lr", type=float, nargs="+", default=[3e-3, 1e-3, 5e-4, 3e-4, 1e-4, 5e-5])
    parser.add_argument("--coarse_batch_size", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--coarse_epochs_2016", type=int, nargs="+", default=[40, 60, 80, 100, 140, 180])
    parser.add_argument("--coarse_epochs_2018", type=int, nargs="+", default=[60, 100, 140, 180, 220])
    parser.add_argument("--coarse_weight_decay", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--weight_decay_followup", type=float, nargs="+", default=[0.0, 1e-5, 1e-4, 5e-4, 1e-3])
    parser.add_argument("--defense_rho", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--defense_pgd_steps", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--defense_sigma", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--defense_shared_top_k", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    baseline_root = os.path.join(args.output_root, "baseline")
    defense_root = os.path.join(args.output_root, "defenses")
    os.makedirs(baseline_root, exist_ok=True)
    os.makedirs(defense_root, exist_ok=True)

    print("== Stage: baseline 2016 coarse ==")
    _run_tune_train(
        data_path=args.data_path_2016,
        dataset_version="2016.10a",
        output_root=baseline_root,
        search_name="baseline2016_coarse",
        lr=args.coarse_lr,
        batch_size=args.coarse_batch_size,
        epochs=args.coarse_epochs_2016,
        weight_decay=args.coarse_weight_decay,
        args=args,
        emit_refine_candidates=True,
    )

    print("\n== Stage: baseline 2016 refine ==")
    refine2016 = _load_refine_candidates(baseline_root, "baseline2016_coarse")
    _run_tune_train(
        data_path=args.data_path_2016,
        dataset_version="2016.10a",
        output_root=baseline_root,
        search_name="baseline2016_refine",
        lr=refine2016["lr"],
        batch_size=refine2016["batch_size"],
        epochs=refine2016["epochs"],
        weight_decay=args.coarse_weight_decay,
        args=args,
        emit_refine_candidates=False,
    )

    print("\n== Stage: baseline 2016 weight decay ==")
    top2016 = _load_top_rows(baseline_root, "baseline2016_refine")[: args.refine_top_k]
    axes2016 = _rows_to_axes(top2016)
    _run_tune_train(
        data_path=args.data_path_2016,
        dataset_version="2016.10a",
        output_root=baseline_root,
        search_name="baseline2016_weight_decay",
        lr=axes2016["lr"],
        batch_size=axes2016["batch_size"],
        epochs=axes2016["epochs"],
        weight_decay=args.weight_decay_followup,
        args=args,
        emit_refine_candidates=True,
    )

    print("\n== Stage: baseline 2018 coarse ==")
    seed2018 = _load_refine_candidates(baseline_root, "baseline2016_weight_decay")
    _run_tune_train(
        data_path=args.data_path_2018,
        dataset_version="2018.01a",
        output_root=baseline_root,
        search_name="baseline2018_coarse",
        lr=seed2018["lr"],
        batch_size=seed2018["batch_size"],
        epochs=args.coarse_epochs_2018,
        weight_decay=seed2018["weight_decay"],
        args=args,
        emit_refine_candidates=True,
    )

    print("\n== Stage: baseline 2018 refine ==")
    refine2018 = _load_refine_candidates(baseline_root, "baseline2018_coarse")
    _run_tune_train(
        data_path=args.data_path_2018,
        dataset_version="2018.01a",
        output_root=baseline_root,
        search_name="baseline2018_refine",
        lr=refine2018["lr"],
        batch_size=refine2018["batch_size"],
        epochs=refine2018["epochs"],
        weight_decay=refine2018["weight_decay"],
        args=args,
        emit_refine_candidates=False,
    )

    print("\n== Stage: defense tuning ==")
    best2018_rows = _load_top_rows(baseline_root, "baseline2018_refine")[: args.defense_shared_top_k]
    defense_axes = _rows_to_axes(best2018_rows)
    cmd = [
        sys.executable,
        TUNE_DEFENSES_PY,
        "--data_path",
        args.data_path_2018,
        "--dataset_version",
        "2018.01a",
        "--num_classes",
        str(args.num_classes_2018),
        "--input_length",
        str(args.input_length_2018),
        "--output_root",
        defense_root,
        "--search_name",
        "defense_tuning",
        "--lr",
        *_stringify(defense_axes["lr"]),
        "--batch_size",
        *_stringify(defense_axes["batch_size"]),
        "--epochs",
        *_stringify(defense_axes["epochs"]),
        "--weight_decay",
        *_stringify(defense_axes["weight_decay"]),
        "--rho",
        *_stringify(args.defense_rho),
        "--pgd_steps",
        *_stringify(args.defense_pgd_steps),
        "--sigma",
        *_stringify(args.defense_sigma),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    _run(cmd)

    summary_rows = []
    for root, search_name in [
        (baseline_root, "baseline2016_coarse"),
        (baseline_root, "baseline2016_refine"),
        (baseline_root, "baseline2016_weight_decay"),
        (baseline_root, "baseline2018_coarse"),
        (baseline_root, "baseline2018_refine"),
    ]:
        rows = _load_top_rows(root, search_name)
        if rows:
            best = rows[0]
            summary_rows.append(
                {
                    "stage": search_name,
                    "best_val_acc": best["best_val_acc"],
                    "lr": best["lr"],
                    "batch_size": best["batch_size"],
                    "epochs": best["epochs"],
                    "weight_decay": best["weight_decay"],
                }
            )

    best_by_defense_path = os.path.join(
        defense_root,
        "defense_tuning",
        "best_by_defense_defense_tuning.json",
    )
    if os.path.isfile(best_by_defense_path):
        best_by_defense = _load_json(best_by_defense_path, "defense_tuning")
        for defense, best in best_by_defense.items():
            summary_rows.append(
                {
                    "stage": "defense_tuning",
                    "defense": defense,
                    "best_val_acc": best["best_val_acc"],
                    "lr": best["lr"],
                    "batch_size": best["batch_size"],
                    "epochs": best["epochs"],
                    "weight_decay": best["weight_decay"],
                    "rho": best.get("rho"),
                    "pgd_steps": best.get("pgd_steps"),
                    "sigma": best.get("sigma"),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_root, "workflow_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWorkflow summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
