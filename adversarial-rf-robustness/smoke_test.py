"""
Smoke test: validates that all pipeline components work end-to-end
with minimal compute (1 batch, 1 epoch, 1 trial).

Usage:
  python smoke_test.py --data_path data/raw/RML2016.10a_dict.pkl
  python smoke_test.py --data_path data/raw/archive-3/GOLD_XYZ_OSC.0001_1024.hdf5 --dataset_version 2018.01a --num_classes 24 --input_length 1024
"""

import os
import sys
import argparse
import tempfile
import shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import RadioMLDataset, get_dataloaders
from models.cnn import RFClassifierCNN
from channels.awgn import AWGNChannel
from channels.rayleigh import RayleighFadingChannel
from channels.cfo import CFOChannel
from channels.composite import CompositeChannel
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def smoke_test(args):
    device = torch.device("cpu")
    passed = 0
    failed = 0
    tmpdir = tempfile.mkdtemp(prefix="smoke_test_")

    def check(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    # 1. Dataset loading
    print("\n1. Dataset loading")
    train_loader = val_loader = test_loader = None

    def test_dataset():
        nonlocal train_loader, val_loader, test_loader
        train_loader, val_loader, test_loader = get_dataloaders(
            data_path=args.data_path,
            dataset_version=args.dataset_version,
            snr_range=None,
            batch_size=32,
            num_workers=0,
            seed=42,
        )
        assert len(train_loader.dataset) > 0, "Empty train set"
        assert len(test_loader.dataset) > 0, "Empty test set"
    check("Load dataset", test_dataset)

    def test_batch_format():
        batch = next(iter(test_loader))
        assert "iq" in batch and "label" in batch and "snr" in batch
        assert batch["iq"].shape[1] == 2, f"Expected 2 I/Q channels, got {batch['iq'].shape[1]}"
        assert batch["iq"].shape[2] == args.input_length, f"Expected length {args.input_length}"
    check("Batch format", test_batch_format)

    def test_snr_labels():
        batch = next(iter(test_loader))
        snrs = batch["snr"].numpy()
        assert snrs.min() >= -30 and snrs.max() <= 40, f"SNR range suspicious: {snrs.min()} to {snrs.max()}"
    check("SNR labels present", test_snr_labels)

    # 2. Model
    print("\n2. Model")
    model = None

    def test_model_create():
        nonlocal model
        model = RFClassifierCNN(num_classes=args.num_classes, input_length=args.input_length).to(device)
        assert model.get_param_count() > 0
    check("Create model", test_model_create)

    def test_model_forward():
        batch = next(iter(test_loader))
        x = batch["iq"].to(device)[:4]
        logits = model(x)
        assert logits.shape == (4, args.num_classes), f"Expected (4, {args.num_classes}), got {logits.shape}"
    check("Forward pass", test_model_forward)

    # 3. Channels
    print("\n3. Channel layers")

    def test_awgn():
        ch = AWGNChannel()
        x = torch.randn(2, 2, args.input_length)
        y = ch(x, snr_db=10.0)
        assert y.shape == x.shape
    check("AWGN channel", test_awgn)

    def test_rayleigh():
        ch = RayleighFadingChannel()
        x = torch.randn(2, 2, args.input_length)
        y = ch(x)
        assert y.shape == x.shape
    check("Rayleigh channel", test_rayleigh)

    def test_composite():
        ch = CompositeChannel([RayleighFadingChannel(), CFOChannel(max_offset=0.01), AWGNChannel()])
        x = torch.randn(2, 2, args.input_length)
        y = ch(x, snr_db=10.0)
        assert y.shape == x.shape
    check("Composite channel", test_composite)

    # 4. Attacks
    print("\n4. Adversarial attacks")

    def test_fgsm():
        batch = next(iter(test_loader))
        x = batch["iq"].to(device)[:4]
        y = batch["label"].to(device)[:4]
        x_adv = fgsm_attack(model, x, y, rho=0.01)
        assert x_adv.shape == x.shape
        assert not torch.allclose(x, x_adv), "FGSM produced no perturbation"
    check("FGSM attack", test_fgsm)

    def test_pgd():
        batch = next(iter(test_loader))
        x = batch["iq"].to(device)[:4]
        y = batch["label"].to(device)[:4]
        x_adv = pgd_attack(model, x, y, rho=0.01, num_steps=3)
        assert x_adv.shape == x.shape
        assert not torch.allclose(x, x_adv), "PGD produced no perturbation"
    check("PGD attack", test_pgd)

    def test_power_constraint():
        batch = next(iter(test_loader))
        x = batch["iq"].to(device)[:4]
        y = batch["label"].to(device)[:4]
        rho = 0.01
        x_adv = pgd_attack(model, x, y, rho=rho, num_steps=5)
        delta = x_adv - x
        x_norm = torch.norm(x.view(x.shape[0], -1), p=2, dim=1)
        d_norm = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
        ratio = (d_norm / (x_norm + 1e-12)).max().item()
        assert ratio <= rho * 1.01, f"Power ratio {ratio:.4f} exceeds budget {rho}"
    check("L2 power constraint", test_power_constraint)

    # 5. SNR filtering (the fixed methodology)
    print("\n5. SNR filtering methodology")

    def test_snr_filtering():
        from evaluate import _filter_by_snr
        batch = next(iter(test_loader))
        x, y = _filter_by_snr(batch, 10.0, device, tolerance=1.0)
        if x is not None:
            # All returned samples should be near SNR=10
            snr_vals = batch["snr"]
            mask = (snr_vals >= 9.0) & (snr_vals <= 11.0)
            assert x.shape[0] == mask.sum().item()
    check("SNR bin filtering", test_snr_filtering)

    # 6. Training (1 epoch)
    print("\n6. Training (1 epoch)")

    def test_training():
        m = RFClassifierCNN(num_classes=args.num_classes, input_length=args.input_length).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=0.001)
        crit = torch.nn.CrossEntropyLoss()
        m.train()
        batch = next(iter(train_loader))
        x = batch["iq"].to(device)
        y = batch["label"].to(device)
        opt.zero_grad()
        logits = m(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        assert loss.item() > 0, "Loss is zero or negative"
    check("1-batch training step", test_training)

    def test_save_load():
        save_path = os.path.join(tmpdir, "test_model.pth")
        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "val_acc": 0.5,
        }, save_path)
        m2 = RFClassifierCNN(num_classes=args.num_classes, input_length=args.input_length)
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        m2.load_state_dict(ckpt["model_state_dict"])
        # Verify outputs match (both in eval mode to disable dropout)
        model.eval()
        m2.eval()
        batch = next(iter(test_loader))
        x = batch["iq"][:2]
        with torch.no_grad():
            out1 = model(x)
            out2 = m2(x)
        assert torch.allclose(out1, out2, atol=1e-5), "Saved/loaded model outputs differ"
    check("Save/load checkpoint", test_save_load)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke Test")
    parser.add_argument("--data_path", type=str, default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--dataset_version", type=str, default="2016.10a")
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--input_length", type=int, default=128)
    args = parser.parse_args()

    success = smoke_test(args)
    sys.exit(0 if success else 1)
