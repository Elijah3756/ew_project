"""
Adversarial transferability analysis.

Tests whether adversarial examples crafted for one model (source)
fool a different model (target). This simulates a more realistic
black-box threat model where the adversary does not have access
to the deployed classifier.

Usage:
  python eval_transferability.py \
    --source_path experiments/results/baseline_cnn/best_model.pth \
    --target_path experiments/results/channel_aug/best_model.pth \
    --data_path data/raw/RML2016.10a_dict.pkl
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import RadioMLDataset, get_dataloaders
from models.cnn import RFClassifierCNN
from channels.awgn import AWGNChannel
from channels.composite import CompositeChannel
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from utils.progress import iter_batches, write_status


def get_device(device_str="auto"):
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path, num_classes, input_length, device):
    model = RFClassifierCNN(num_classes=num_classes, input_length=input_length).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def eval_transferability(source_model, target_model, test_loader, device,
                          snr_db=10, rho=0.01, attack_type="pgd", pgd_steps=10,
                          num_trials=10, progress=True, progress_file=None):
    """
    Generate adversarial examples using source_model, evaluate on target_model.

    Returns dict with:
      - source_clean_acc, source_robust_acc, source_asr (white-box)
      - target_clean_acc, target_robust_acc, target_asr (transfer / black-box)
      - transfer_rate: fraction of successful white-box attacks that also fool the target
    """
    attack_fn = pgd_attack if attack_type == "pgd" else fgsm_attack

    trial_results = []
    t0 = time.time()

    for trial in range(num_trials):
        src_correct_clean, src_correct_adv = 0, 0
        tgt_correct_clean, tgt_correct_adv = 0, 0
        src_fooled, tgt_fooled, both_fooled = 0, 0, 0
        src_fooled_tgt_correct = 0
        total = 0
        desc = f"transfer SNR{snr_db:.0f} t{trial + 1}/{num_trials}"
        for batch in iter_batches(test_loader, desc, progress):
            x = batch["iq"].to(device)
            y = batch["label"].to(device)
            snr_mask = (batch["snr"] >= snr_db - 1) & (batch["snr"] <= snr_db + 1)
            if snr_mask.sum() == 0:
                continue
            x = x[snr_mask]
            y = y[snr_mask]

            # Clean predictions -- native samples, no additional AWGN
            with torch.no_grad():
                src_preds_clean = source_model(x).argmax(1)
                tgt_preds_clean = target_model(x).argmax(1)

            src_correct_clean += (src_preds_clean == y).sum().item()
            tgt_correct_clean += (tgt_preds_clean == y).sum().item()

            # Generate adversarial examples using SOURCE model (no channel)
            attack_kwargs = {"rho": rho}
            if attack_type == "pgd":
                attack_kwargs["num_steps"] = pgd_steps

            x_adv = attack_fn(source_model, x, y, **attack_kwargs)

            # Evaluate adversarial examples on BOTH models (native, no double-noising)
            with torch.no_grad():
                src_preds_adv = source_model(x_adv).argmax(1)
                tgt_preds_adv = target_model(x_adv).argmax(1)

            src_correct_adv += (src_preds_adv == y).sum().item()
            tgt_correct_adv += (tgt_preds_adv == y).sum().item()

            # Attack success: correctly classified clean, misclassified adversarial
            src_correct_mask = (src_preds_clean == y)
            tgt_correct_mask = (tgt_preds_clean == y)

            src_fooled_mask = (src_preds_adv != y) & src_correct_mask
            tgt_fooled_mask = (tgt_preds_adv != y) & tgt_correct_mask

            src_fooled += src_fooled_mask.sum().item()
            tgt_fooled += tgt_fooled_mask.sum().item()
            # Transfer: fooled source AND fooled target, conditioned on BOTH
            # source and target correctly classifying the clean sample.
            # This avoids inflating transfer rate when target was already wrong.
            both_correct_mask = src_correct_mask & tgt_correct_mask
            both_fooled += ((src_preds_adv != y) & both_correct_mask & (tgt_preds_adv != y)).sum().item()
            # Count source-fooled samples where target was also correct on clean
            src_fooled_tgt_correct += ((src_preds_adv != y) & src_correct_mask & tgt_correct_mask).sum().item()
            total += y.size(0)

        if total == 0:
            continue

        src_clean = src_correct_clean / total
        tgt_clean = tgt_correct_clean / total
        src_asr = src_fooled / max(src_correct_clean, 1)
        tgt_asr = tgt_fooled / max(tgt_correct_clean, 1)
        # Transfer rate: of source-fooled samples where target was also correct
        # on clean, what fraction also fooled the target?
        transfer_rate = both_fooled / max(src_fooled_tgt_correct, 1)

        trial_results.append({
            "source_clean_acc": src_clean,
            "source_robust_acc": src_correct_adv / total,
            "source_asr": src_asr,
            "target_clean_acc": tgt_clean,
            "target_robust_acc": tgt_correct_adv / total,
            "target_asr": tgt_asr,
            "transfer_rate": transfer_rate,
        })

        write_status(
            progress_file,
            mode="transfer_trial",
            snr_db=float(snr_db),
            trial_index=trial + 1,
            num_trials=num_trials,
            elapsed_s=round(time.time() - t0, 1),
        )

    if not trial_results:
        return {}

    # Aggregate with 95% CI
    n_t = len(trial_results)
    ci_mult = stats.t.ppf(0.975, df=max(n_t - 1, 1)) if n_t > 1 else 0

    result = {}
    for key in trial_results[0]:
        vals = [r[key] for r in trial_results]
        result[key] = np.mean(vals)
        result[f"{key}_std"] = np.std(vals, ddof=1) if n_t > 1 else 0.0
        if n_t > 1:
            result[f"{key}_ci95"] = ci_mult * np.std(vals, ddof=1) / np.sqrt(n_t)
        else:
            result[f"{key}_ci95"] = 0

    return result


def main():
    parser = argparse.ArgumentParser(description="Adversarial Transferability Analysis")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to source model checkpoint (adversary uses this)")
    parser.add_argument("--target_path", type=str, nargs="+", required=True,
                        help="Path(s) to target model checkpoint(s)")
    parser.add_argument("--target_names", type=str, nargs="+", default=None,
                        help="Names for target models (for output)")
    parser.add_argument("--source_name", type=str, default="source")
    parser.add_argument("--data_path", type=str, default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--dataset_version", type=str, default="2016.10a",
                        choices=["2016.10a", "2018.01a"])
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--input_length", type=int, default=128)
    parser.add_argument("--snr", type=float, nargs="+", default=[0, 10])
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--attack", type=str, default="pgd", choices=["pgd", "fgsm"])
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm batch bars and eval_progress.json")
    parser.add_argument("--progress-file", type=str, default=None,
                        help="JSON status path (default: <output_dir>/transferability_progress.json)")
    args = parser.parse_args()

    show_progress = not args.no_progress
    progress_file = None
    if show_progress:
        progress_file = args.progress_file or os.path.join(
            args.output_dir, "transferability_progress.json")

    device = get_device(args.device)
    print(f"Device: {device}")

    # Load source model
    source_model = load_model(args.source_path, args.num_classes, args.input_length, device)
    print(f"Source model: {args.source_name} ({args.source_path})")

    # Load target models
    target_names = args.target_names or [f"target_{i}" for i in range(len(args.target_path))]
    targets = []
    for name, path in zip(target_names, args.target_path):
        m = load_model(path, args.num_classes, args.input_length, device)
        targets.append((name, m))
        print(f"Target model: {name} ({path})")

    # Load test data
    _, _, test_loader = get_dataloaders(
        data_path=args.data_path, dataset_version=args.dataset_version,
        snr_range=None, batch_size=256, num_workers=0, seed=42)  # Load all SNRs; filtering done in eval
    print(f"Test set: {len(test_loader.dataset):,} samples")
    if progress_file:
        print(f"Progress status file: {progress_file}")
    write_status(progress_file, mode="started", script="eval_transferability")

    # Run transferability analysis
    all_rows = []
    for snr in args.snr:
        print(f"\n{'='*60}")
        print(f"SNR = {snr} dB | {args.attack.upper()} | rho = {args.rho}")
        print(f"{'='*60}")

        for tgt_name, tgt_model in targets:
            print(f"\n  Source: {args.source_name} -> Target: {tgt_name}")
            result = eval_transferability(
                source_model, tgt_model, test_loader, device,
                snr_db=snr, rho=args.rho, attack_type=args.attack,
                pgd_steps=args.pgd_steps, num_trials=args.num_trials,
                progress=show_progress,
                progress_file=progress_file)

            if result:
                print(f"    White-box ASR (source): {result['source_asr']:.1%}")
                print(f"    Black-box ASR (target): {result['target_asr']:.1%}")
                print(f"    Transfer rate:          {result['transfer_rate']:.1%}")
                all_rows.append({
                    "snr": snr,
                    "source": args.source_name,
                    "target": tgt_name,
                    **result,
                })
                write_status(
                    progress_file,
                    mode="transfer_pair_done",
                    snr_db=float(snr),
                    source=args.source_name,
                    target=tgt_name,
                )

    # Save results
    if all_rows:
        df = pd.DataFrame(all_rows)
        out_path = os.path.join(args.output_dir, "transferability_analysis.csv")
        df.to_csv(out_path, index=False)
        write_status(progress_file, mode="complete", script="eval_transferability", rows=len(all_rows))
        print(f"\nResults saved to {out_path}")

        # Pretty summary table
        print(f"\n{'='*70}")
        print("TRANSFERABILITY SUMMARY")
        print(f"{'='*70}")
        print(f"{'SNR':>5} {'Source':>12} {'Target':>15} {'WB-ASR':>8} {'BB-ASR':>8} {'Transfer':>10}")
        print("-" * 70)
        for _, row in df.iterrows():
            print(f"{row['snr']:>5.0f} {row['source']:>12} {row['target']:>15} "
                  f"{row['source_asr']:>7.1%} {row['target_asr']:>7.1%} "
                  f"{row['transfer_rate']:>9.1%}")


if __name__ == "__main__":
    main()
