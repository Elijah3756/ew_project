#!/bin/bash
# =============================================================
# Master experiment runner
# Runs all phases sequentially on GPU
# Usage: bash run_all.sh
# =============================================================

set -e

DATA_PATH="data/raw/RML2016.10a_dict.pkl"
# Auto-detect device: uses CUDA if available, then MPS (Apple Silicon), then CPU
DEVICE="auto"
SEED=42

echo "=============================================="
echo " Phase 0: Baseline CNN Training"
echo "=============================================="
python train.py \
  --data_path $DATA_PATH \
  --dataset_version 2016.10a \
  --epochs 60 \
  --batch_size 256 \
  --lr 0.001 \
  --seed $SEED \
  --save_dir experiments/results/baseline_cnn \
  --device $DEVICE

echo ""
echo "=============================================="
echo " Phase 0: Channel-Augmented CNN Training"
echo "=============================================="
python train.py \
  --data_path $DATA_PATH \
  --dataset_version 2016.10a \
  --epochs 60 \
  --batch_size 256 \
  --lr 0.001 \
  --seed $SEED \
  --save_dir experiments/results/channel_aug_cnn \
  --channel_aug \
  --device $DEVICE

echo ""
echo "=============================================="
echo " Phase 1: Clean Accuracy vs SNR (all channels)"
echo "=============================================="
# Baseline model
python evaluate.py \
  --model_path experiments/results/baseline_cnn/best_model.pth \
  --data_path $DATA_PATH \
  --eval_mode clean_snr \
  --output_dir experiments/results/baseline_cnn \
  --device $DEVICE \
  --seed $SEED

# Channel-augmented model
python evaluate.py \
  --model_path experiments/results/channel_aug_cnn/best_model.pth \
  --data_path $DATA_PATH \
  --eval_mode clean_snr \
  --output_dir experiments/results/channel_aug_cnn \
  --device $DEVICE \
  --seed $SEED

echo ""
echo "=============================================="
echo " Phase 2: FGSM Attack vs SNR"
echo "=============================================="
for CHANNEL in awgn rayleigh_awgn rayleigh_cfo_awgn; do
  echo "--- Channel: $CHANNEL ---"
  python evaluate.py \
    --model_path experiments/results/baseline_cnn/best_model.pth \
    --data_path $DATA_PATH \
    --eval_mode attack_snr \
    --attack_type fgsm \
    --channel_type $CHANNEL \
    --rho 0.01 \
    --output_dir experiments/results/baseline_cnn \
    --device $DEVICE \
    --seed $SEED \
  --num_trials 10
done

echo ""
echo "=============================================="
echo " Phase 2: PGD Attack vs SNR"
echo "=============================================="
for CHANNEL in awgn rayleigh_awgn rayleigh_cfo_awgn; do
  echo "--- Channel: $CHANNEL ---"
  python evaluate.py \
    --model_path experiments/results/baseline_cnn/best_model.pth \
    --data_path $DATA_PATH \
    --eval_mode attack_snr \
    --attack_type pgd \
    --channel_type $CHANNEL \
    --rho 0.01 \
    --pgd_steps 10 \
    --output_dir experiments/results/baseline_cnn \
    --device $DEVICE \
    --seed $SEED \
    --num_trials 10
done

echo ""
echo "=============================================="
echo " Phase 2: Attack vs Perturbation Budget"
echo "=============================================="
for ATTACK in fgsm pgd; do
  echo "--- Attack: $ATTACK ---"
  python evaluate.py \
    --model_path experiments/results/baseline_cnn/best_model.pth \
    --data_path $DATA_PATH \
    --eval_mode attack_budget \
    --attack_type $ATTACK \
    --channel_type awgn \
    --pgd_steps 10 \
    --output_dir experiments/results/baseline_cnn \
    --device $DEVICE \
    --seed $SEED \
    --num_trials 10
done

echo ""
echo "=============================================="
echo " Phase 3: Defense Training & Evaluation"
echo "=============================================="
python train_defenses.py \
  --data_path $DATA_PATH \
  --epochs 60 \
  --batch_size 256 \
  --lr 0.001 \
  --seed $SEED \
  --rho 0.01 \
  --pgd_steps 5 \
  --sigma 0.01 \
  --output_dir experiments/results \
  --eval_snr 0 10 \
  --device $DEVICE

echo ""
echo "=============================================="
echo " Phase 4: Per-Class Vulnerability Analysis"
echo "=============================================="
python eval_per_class.py \
  --model_path experiments/results/baseline_cnn/best_model.pth \
  --data_path $DATA_PATH \
  --snr 0 10 18 \
  --rho 0.01 \
  --attack pgd \
  --pgd_steps 10 \
  --num_trials 10 \
  --output_dir experiments/results \
  --device $DEVICE

echo ""
echo "=============================================="
echo " Generate All Figures"
echo "=============================================="
python plot_results.py \
  --results_dir experiments/results \
  --output_dir experiments/figures

echo ""
echo "=============================================="
echo " ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Results: experiments/results/"
echo "Figures: experiments/figures/"
