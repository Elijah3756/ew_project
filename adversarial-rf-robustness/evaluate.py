"""
Comprehensive evaluation script for adversarial robustness experiments.

Runs:
  1. Clean accuracy vs SNR (baseline)
  2. Channel-impaired accuracy vs SNR
  3. Attack success vs SNR (FGSM, PGD)
  4. Attack success vs perturbation budget
  5. Defense comparison

SNR Methodology:
  RadioML datasets contain samples generated at specific SNR levels. Each sample
  already has AWGN baked in at its labeled SNR. To avoid double-noising, this
  script FILTERS samples by their dataset SNR label rather than adding new AWGN.

  For additional channel impairments (Rayleigh fading, CFO), only the non-AWGN
  effects are applied on top of the native samples. This is controlled via the
  build_channel_native() function which strips AWGN from channel chains.

Usage:
  python evaluate.py --model_path experiments/results/baseline_cnn/best_model.pth \
                     --eval_mode clean_snr
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import RadioMLDataset, get_dataloaders
from utils.progress import iter_batches, write_status
from models.cnn import RFClassifierCNN
from channels.awgn import AWGNChannel
from channels.rayleigh import RayleighFadingChannel
from channels.cfo import CFOChannel
from channels.composite import CompositeChannel
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_path, num_classes=11, input_length=128, device="cpu"):
    """Load a trained model from checkpoint."""
    model = RFClassifierCNN(num_classes=num_classes, input_length=input_length)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')}, "
          f"val_acc={checkpoint.get('val_acc', '?'):.4f})")
    return model


def build_channel(channel_type: str, device: str = "cpu"):
    """
    Build full channel configuration by name (includes AWGN).
    Used during TRAINING where we want to add channel noise.
    """
    channels = {
        "clean": None,
        "awgn": CompositeChannel([AWGNChannel()]),
        "rayleigh_awgn": CompositeChannel([
            RayleighFadingChannel(),
            AWGNChannel(),
        ]),
        "rayleigh_cfo_awgn": CompositeChannel([
            RayleighFadingChannel(),
            CFOChannel(max_offset=0.01),
            AWGNChannel(),
        ]),
    }
    ch = channels.get(channel_type)
    if ch is not None:
        ch = ch.to(device)
    return ch


def build_channel_native(channel_type: str, device: str = "cpu"):
    """
    Build channel configuration WITHOUT AWGN for use during evaluation.

    RadioML samples already have noise at their native SNR. When evaluating
    accuracy vs SNR, we filter by dataset SNR label and do NOT add AWGN.
    Only non-AWGN impairments (fading, CFO) are applied on top.

    KNOWN LIMITATION: RadioML samples contain baked-in AWGN at their labeled SNR.
    When we apply fading (h * x), we are computing h * (x_clean + n_native) instead
    of the physically correct (h * x_clean) + n_thermal. This means:
      - Fading does not degrade SNR as it would in reality (signal and noise
        are attenuated equally).
      - This makes our fading-channel results OPTIMISTIC for the classifier
        (fading appears less harmful than it truly is).
      - For a fully correct evaluation, noiseless source signals would be needed,
        with receiver thermal noise added AFTER fading.
    Despite this limitation, the RELATIVE comparison between clean and adversarial
    performance under fading remains informative, as both are subject to the same
    approximation.

    Returns:
        channel: nn.Module or None (None means no additional impairments needed)
    """
    channels = {
        "clean": None,
        "awgn": None,  # AWGN already in dataset -- no additional channel needed
        "rayleigh_awgn": CompositeChannel([
            RayleighFadingChannel(),
        ]),
        "rayleigh_cfo_awgn": CompositeChannel([
            RayleighFadingChannel(),
            CFOChannel(max_offset=0.01),
        ]),
    }
    ch = channels.get(channel_type)
    if ch is not None:
        ch = ch.to(device)
    return ch


def _filter_by_snr(batch, snr_target, device, tolerance=1.0):
    """
    Filter a batch to only include samples at a specific SNR bin.

    RadioML 2016.10a uses 2 dB steps, 2018.01a uses 2 dB steps.
    A tolerance of 1.0 dB captures exact matches for integer SNR bins.

    Args:
        batch: Dict with 'iq', 'label', 'snr' tensors.
        snr_target: Target SNR in dB.
        device: Torch device.
        tolerance: Half-width of SNR bin filter (default 1.0 dB).

    Returns:
        x, y: Filtered I/Q tensor and labels on device. May be empty.
    """
    x = batch["iq"].to(device)
    y = batch["label"].to(device)
    snr_vals = batch["snr"]  # keep on CPU for comparison

    snr_mask = (snr_vals >= snr_target - tolerance) & (snr_vals <= snr_target + tolerance)

    if snr_mask.sum() == 0:
        return None, None

    return x[snr_mask], y[snr_mask]


def _compute_ci(values, confidence=0.95):
    """Compute mean, sample std (ddof=1), and confidence interval half-width."""
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0
    if n > 1:
        ci = stats.t.ppf(0.5 + confidence / 2, df=n - 1) * std / np.sqrt(n)
    else:
        ci = 0
    return mean, std, ci


# ============================================================
# Evaluation Mode 1: Clean Accuracy vs SNR
# ============================================================
def eval_clean_snr(model, test_loader, device, snr_values, channel_types=None, num_trials=10,
                   progress=True, progress_file=None):
    """
    Evaluate clean accuracy across SNR for multiple channel configs.

    Methodology: Samples are FILTERED by their native dataset SNR label.
    No AWGN is added (it's already baked into the samples). For non-AWGN
    channels (Rayleigh, CFO), only those impairments are applied.

    Multiple trials are run to average over stochastic channel realizations
    (relevant for fading channels; for AWGN-only this is deterministic since
    no channel is applied).
    """
    if channel_types is None:
        channel_types = ["awgn", "rayleigh_awgn", "rayleigh_cfo_awgn"]

    all_results = {}
    t0 = time.time()

    for ch_type in channel_types:
        print(f"\n  Channel: {ch_type}")
        channel = build_channel_native(ch_type, device)
        snr_accs = {}

        for snr_i, snr in enumerate(snr_values):
            trial_accs = []
            for trial in range(num_trials):
                correct, total = 0, 0
                desc = f"{ch_type} SNR{snr:+d}dB t{trial + 1}/{num_trials}"
                for batch in iter_batches(test_loader, desc, progress):
                    x, y = _filter_by_snr(batch, snr, device)
                    if x is None:
                        continue

                    # Apply non-AWGN channel impairments if any
                    if channel is not None:
                        x_ch = channel(x)
                    else:
                        x_ch = x  # Native samples at this SNR, no additional processing

                    with torch.no_grad():
                        logits = model(x_ch)
                    correct += (logits.argmax(1) == y).sum().item()
                    total += y.size(0)

                if total == 0:
                    continue
                trial_accs.append(correct / total)

            if not trial_accs:
                print(f"    SNR {snr:+3d} dB: NO SAMPLES")
                continue

            mean_acc, std_acc, ci95 = _compute_ci(trial_accs)
            snr_accs[snr] = {"mean": mean_acc, "std": std_acc, "ci95": ci95}
            print(f"    SNR {snr:+3d} dB: {mean_acc:.4f} +/- {std_acc:.4f} (CI95: {ci95:.4f})")
            write_status(
                progress_file,
                mode="clean_snr",
                channel=ch_type,
                snr_db=float(snr),
                snr_index=snr_i + 1,
                total_snrs=len(snr_values),
                accuracy_mean=float(mean_acc),
                elapsed_s=round(time.time() - t0, 1),
            )

        all_results[ch_type] = snr_accs

    return all_results


# ============================================================
# Evaluation Mode 2: Attack Success vs SNR
# ============================================================
def eval_attack_snr(model, test_loader, device, snr_values, attack_type="pgd",
                    rho=0.01, channel_type="awgn", pgd_steps=10, num_trials=10,
                    freeze_channel=False, progress=True, progress_file=None):
    """
    Evaluate attack success rate and robust accuracy across SNR.

    Methodology: For each target SNR, only samples at that native SNR bin
    are used. No additional AWGN is applied. Non-AWGN channel effects
    (fading, CFO) are applied as additional impairments.

    The attack crafts perturbations using the native-channel (no AWGN)
    configuration, then evaluation uses an independent channel realization.
    Multiple trials average over stochastic channel realizations.

    NOTE ON FADING CHANNELS: Because RadioML samples contain baked-in AWGN,
    the fading channel applies to signal+noise jointly (h*(x+n)) rather than
    the physically correct (h*x)+n. This is a known approximation that makes
    fading-channel results optimistic. See build_channel_native() for details.
    """
    channel = build_channel_native(channel_type, device)
    attack_fn = pgd_attack if attack_type == "pgd" else fgsm_attack

    results = {}
    t0 = time.time()

    for snr_i, snr in enumerate(snr_values):
        trial_results = []
        for trial in range(num_trials):
            correct_clean, correct_adv, total, fooled = 0, 0, 0, 0
            desc = f"attack({attack_type}) {channel_type} SNR{snr:+d} t{trial + 1}/{num_trials}"
            for batch in iter_batches(test_loader, desc, progress):
                x, y = _filter_by_snr(batch, snr, device)
                if x is None:
                    continue

                # Clean prediction (with non-AWGN channel effects if any)
                if channel is not None:
                    x_ch = channel(x)
                else:
                    x_ch = x

                with torch.no_grad():
                    logits_clean = model(x_ch)
                preds_clean = logits_clean.argmax(1)
                correct_clean += (preds_clean == y).sum().item()

                # Adversarial attack -- channel passed to attack for
                # gradient computation through channel impairments
                attack_kwargs = {"rho": rho}
                if attack_type == "pgd":
                    attack_kwargs["num_steps"] = pgd_steps
                    attack_kwargs["freeze_channel"] = freeze_channel

                # Pass channel (fading-only) so attack is channel-aware.
                # snr_db is not needed since we don't have AWGN in channel.
                x_adv = attack_fn(model, x, y, channel=channel, **attack_kwargs)

                # Adversarial prediction (new channel realization)
                # NOTE: Attack and evaluation use INDEPENDENT channel realizations.
                # When freeze_channel=False (default): PGD steps also use independent
                # realizations (no-CSI stochastic attack).
                # When freeze_channel=True: PGD steps use a fixed realization (perfect
                # CSI attack), but evaluation still uses an independent realization,
                # modeling the scenario where the adversary knows the attack-time
                # channel but not the evaluation-time channel.
                #
                # ADDITIONAL NOTE: Because RadioML samples contain baked-in AWGN,
                # the fading channel applies to signal+noise jointly (h*(x+n) rather
                # than h*x+n). This is a known approximation; see build_channel_native().
                if channel is not None:
                    x_adv_ch = channel(x_adv)
                else:
                    x_adv_ch = x_adv

                with torch.no_grad():
                    logits_adv = model(x_adv_ch)
                preds_adv = logits_adv.argmax(1)
                correct_adv += (preds_adv == y).sum().item()

                # Attack success (on correctly classified samples)
                correct_mask = (preds_clean == y)
                fooled += ((preds_adv != y) & correct_mask).sum().item()
                total += y.size(0)

            if total == 0:
                continue

            clean_acc = correct_clean / total
            robust_acc = correct_adv / total
            asr = fooled / max(correct_clean, 1)
            trial_results.append({
                "clean_acc": clean_acc,
                "robust_acc": robust_acc,
                "attack_success_rate": asr,
            })

        if not trial_results:
            print(f"  SNR {snr:+3d} dB | NO SAMPLES")
            continue

        clean_mean, clean_std, clean_ci = _compute_ci([r["clean_acc"] for r in trial_results])
        robust_mean, robust_std, robust_ci = _compute_ci([r["robust_acc"] for r in trial_results])
        asr_mean, asr_std, asr_ci = _compute_ci([r["attack_success_rate"] for r in trial_results])

        results[snr] = {
            "clean_acc": clean_mean,
            "robust_acc": robust_mean,
            "attack_success_rate": asr_mean,
            "clean_acc_std": clean_std,
            "robust_acc_std": robust_std,
            "asr_std": asr_std,
            "clean_ci95": clean_ci,
            "robust_ci95": robust_ci,
            "asr_ci95": asr_ci,
        }
        print(f"  SNR {snr:+3d} dB | Clean: {clean_mean:.4f} | "
              f"Robust: {robust_mean:.4f} | ASR: {asr_mean:.4f}")
        write_status(
            progress_file,
            mode="attack_snr",
            attack=attack_type,
            channel=channel_type,
            snr_db=float(snr),
            snr_index=snr_i + 1,
            total_snrs=len(snr_values),
            clean_acc=float(clean_mean),
            robust_acc=float(robust_mean),
            asr=float(asr_mean),
            elapsed_s=round(time.time() - t0, 1),
        )

    return results


# ============================================================
# Evaluation Mode 3: Attack Success vs Perturbation Budget
# ============================================================
def eval_attack_budget(model, test_loader, device, rho_values, snr_values_fixed,
                       attack_type="pgd", channel_type="awgn", pgd_steps=10, num_trials=10,
                       freeze_channel=False, progress=True, progress_file=None):
    """
    Evaluate attack success across perturbation budgets at fixed SNR values.

    Methodology: Same SNR-filtering approach as eval_attack_snr. Samples
    at each fixed SNR are filtered from the dataset. No additional AWGN.
    """
    channel = build_channel_native(channel_type, device)
    attack_fn = pgd_attack if attack_type == "pgd" else fgsm_attack

    results = {}
    t0 = time.time()

    for snr in snr_values_fixed:
        results[snr] = {}
        print(f"\n  SNR = {snr} dB:")

        for rho_i, rho in enumerate(rho_values):
            trial_asrs = []
            trial_robust = []

            for trial in range(num_trials):
                correct_clean, correct_adv, total, fooled = 0, 0, 0, 0
                desc = (
                    f"budget({attack_type}) SNR{snr} rho{rho * 100:.1f}% "
                    f"t{trial + 1}/{num_trials}"
                )
                for batch in iter_batches(test_loader, desc, progress):
                    x, y = _filter_by_snr(batch, snr, device)
                    if x is None:
                        continue

                    # Clean prediction
                    if channel is not None:
                        x_ch = channel(x)
                    else:
                        x_ch = x

                    with torch.no_grad():
                        logits_clean = model(x_ch)
                    preds_clean = logits_clean.argmax(1)
                    correct_clean += (preds_clean == y).sum().item()

                    # Attack
                    attack_kwargs = {"rho": rho}
                    if attack_type == "pgd":
                        attack_kwargs["num_steps"] = pgd_steps
                        attack_kwargs["freeze_channel"] = freeze_channel

                    x_adv = attack_fn(model, x, y, channel=channel, **attack_kwargs)

                    # Adversarial prediction (new channel realization)
                    if channel is not None:
                        x_adv_ch = channel(x_adv)
                    else:
                        x_adv_ch = x_adv

                    with torch.no_grad():
                        logits_adv = model(x_adv_ch)
                    preds_adv = logits_adv.argmax(1)
                    correct_adv += (preds_adv == y).sum().item()

                    correct_mask = (preds_clean == y)
                    fooled += ((preds_adv != y) & correct_mask).sum().item()
                    total += y.size(0)

                if total == 0:
                    continue
                trial_asrs.append(fooled / max(correct_clean, 1))
                trial_robust.append(correct_adv / total)

            if not trial_asrs:
                print(f"    rho={rho*100:.1f}% | NO SAMPLES")
                continue

            asr_mean, asr_std, _ = _compute_ci(trial_asrs)
            rob_mean, rob_std, _ = _compute_ci(trial_robust)

            results[snr][rho] = {
                "attack_success_rate": asr_mean,
                "robust_accuracy": rob_mean,
                "asr_std": asr_std,
                "robust_std": rob_std,
            }
            print(f"    rho={rho*100:.1f}% | ASR: {asr_mean:.4f} | "
                  f"Robust Acc: {rob_mean:.4f}")
            write_status(
                progress_file,
                mode="attack_budget",
                attack=attack_type,
                channel=channel_type,
                snr_db=float(snr),
                rho=float(rho),
                rho_index=rho_i + 1,
                total_rhos=len(rho_values),
                asr=float(asr_mean),
                robust_acc=float(rob_mean),
                elapsed_s=round(time.time() - t0, 1),
            )

    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate RF Adversarial Robustness")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--dataset_version", type=str, default="2016.10a")
    parser.add_argument("--eval_mode", type=str, required=True,
                        choices=["clean_snr", "attack_snr", "attack_budget", "full"])
    parser.add_argument("--attack_type", type=str, default="pgd", choices=["fgsm", "pgd"])
    parser.add_argument("--channel_type", type=str, default="awgn",
                        choices=["clean", "awgn", "rayleigh_awgn", "rayleigh_cfo_awgn"])
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="experiments/results/eval")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--input_length", type=int, default=128)
    parser.add_argument("--snr_values", type=int, nargs="+", default=None,
                        help="SNR values to evaluate (default: -10 to 18 in 2dB steps)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to training config JSON (honors snr_range, batch_size, etc.)")
    parser.add_argument("--freeze-channel", action="store_true", default=False,
                        help="For PGD attacks: freeze channel realization during optimization (perfect CSI)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm batch bars and eval_progress.json status snapshots")
    parser.add_argument("--progress-file", type=str, default=None,
                        help="Where to write JSON status (default: <output_dir>/eval_progress.json)")
    args = parser.parse_args()

    # Load config if provided and apply missing args
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
        print(f"Loaded config from {args.config}")
        # Config can set defaults for unspecified args
        if args.batch_size == 256 and "batch_size" in config:
            args.batch_size = config["batch_size"]
        if args.num_classes == 11 and "num_classes" in config:
            args.num_classes = config["num_classes"]
        if args.input_length == 128 and "input_length" in config:
            args.input_length = config["input_length"]
        if args.data_path == "data/raw/RML2016.10a_dict.pkl" and "data_path" in config:
            args.data_path = config["data_path"]
        if args.dataset_version == "2016.10a" and "dataset_version" in config:
            args.dataset_version = config["dataset_version"]

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    show_progress = not args.no_progress
    progress_file = None
    if show_progress:
        progress_file = args.progress_file or os.path.join(args.output_dir, "eval_progress.json")

    # Load model
    model = load_model(args.model_path, args.num_classes, args.input_length, device)

    # Load test data -- load ALL SNR ranges since we filter per-SNR during eval
    _, _, test_loader = get_dataloaders(
        data_path=args.data_path,
        dataset_version=args.dataset_version,
        snr_range=None,  # Load all SNRs, we filter in eval functions
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
    )
    print(f"Test samples: {len(test_loader.dataset):,}")
    if progress_file:
        print(f"Progress status file: {progress_file}")
    write_status(progress_file, mode="started", eval_mode=args.eval_mode, device=str(device))

    # SNR values to evaluate (RadioML uses 2dB steps)
    if args.snr_values is not None:
        snr_values = args.snr_values
    elif args.dataset_version == "2018.01a":
        snr_values = list(range(-20, 32, 2))  # 2018 goes up to +30 dB
    else:
        snr_values = list(range(-20, 20, 2))  # 2016 goes -20 to +18
    rho_values = [0.005, 0.01, 0.02, 0.05]

    # ---- Clean SNR ----
    if args.eval_mode in ["clean_snr", "full"]:
        print("\n" + "=" * 60)
        print("CLEAN ACCURACY vs SNR (native dataset SNR, no double-noising)")
        print("=" * 60)
        clean_results = eval_clean_snr(
            model, test_loader, device, snr_values,
            channel_types=["awgn", "rayleigh_awgn", "rayleigh_cfo_awgn"],
            num_trials=args.num_trials,
            progress=show_progress,
            progress_file=progress_file,
        )
        # Save
        for ch_type, snr_accs in clean_results.items():
            df = pd.DataFrame([
                {"snr": snr, "accuracy": v["mean"], "accuracy_std": v["std"],
                 "accuracy_ci95": v["ci95"]}
                for snr, v in snr_accs.items()
            ])
            df.to_csv(os.path.join(args.output_dir, f"clean_snr_{ch_type}.csv"), index=False)
        print(f"\nSaved clean SNR results to {args.output_dir}/")

    # ---- Attack vs SNR ----
    if args.eval_mode in ["attack_snr", "full"]:
        print("\n" + "=" * 60)
        print(f"ATTACK ({args.attack_type.upper()}) vs SNR  [rho={args.rho*100:.1f}%, channel={args.channel_type}]")
        print("=" * 60)
        attack_results = eval_attack_snr(
            model, test_loader, device, snr_values,
            attack_type=args.attack_type,
            rho=args.rho,
            channel_type=args.channel_type,
            pgd_steps=args.pgd_steps,
            num_trials=args.num_trials,
            freeze_channel=args.freeze_channel,
            progress=show_progress,
            progress_file=progress_file,
        )
        df = pd.DataFrame([
            {"snr": snr, **vals}
            for snr, vals in attack_results.items()
        ])
        fname = f"attack_snr_{args.attack_type}_{args.channel_type}_rho{args.rho}.csv"
        df.to_csv(os.path.join(args.output_dir, fname), index=False)
        print(f"\nSaved to {args.output_dir}/{fname}")

    # ---- Attack vs Budget ----
    if args.eval_mode in ["attack_budget", "full"]:
        print("\n" + "=" * 60)
        print(f"ATTACK ({args.attack_type.upper()}) vs PERTURBATION BUDGET  [channel={args.channel_type}]")
        print("=" * 60)
        budget_results = eval_attack_budget(
            model, test_loader, device, rho_values,
            snr_values_fixed=[0, 10],
            attack_type=args.attack_type,
            channel_type=args.channel_type,
            pgd_steps=args.pgd_steps,
            num_trials=args.num_trials,
            freeze_channel=args.freeze_channel,
            progress=show_progress,
            progress_file=progress_file,
        )
        rows = []
        for snr, rho_dict in budget_results.items():
            for rho, vals in rho_dict.items():
                rows.append({"snr": snr, "rho": rho, **vals})
        df = pd.DataFrame(rows)
        fname = f"attack_budget_{args.attack_type}_{args.channel_type}.csv"
        df.to_csv(os.path.join(args.output_dir, fname), index=False)
        print(f"\nSaved to {args.output_dir}/{fname}")

    write_status(progress_file, mode="complete", eval_mode=args.eval_mode)
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
