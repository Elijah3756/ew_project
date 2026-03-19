#!/bin/bash
# =============================================================
# Experiment runner for RadioML 2018.01A dataset
# 24 modulations, 1024 I/Q samples per signal
# Usage: bash run_2018.sh
# =============================================================

set -e

DATA_PATH="data/raw/archive-3/GOLD_XYZ_OSC.0001_1024.hdf5"
DATASET_VERSION="2018.01a"
NUM_CLASSES=24
INPUT_LENGTH=1024
DEVICE="auto"
SEED=42
RESULTS_DIR="experiments/results_2018"

echo "=============================================="
echo " RadioML 2018.01A Experiment Pipeline"
echo " 24 modulations, 1024 I/Q samples"
echo "=============================================="

echo ""
echo "=============================================="
echo " Phase 0: Baseline CNN Training (2018)"
echo "=============================================="
python train.py \
  --data_path $DATA_PATH \
  --dataset_version $DATASET_VERSION \
  --num_classes $NUM_CLASSES \
  --input_length $INPUT_LENGTH \
  --epochs 60 \
  --batch_size 256 \
  --lr 0.001 \
  --seed $SEED \
  --save_dir ${RESULTS_DIR}/baseline_cnn \
  --device $DEVICE

echo ""
echo "=============================================="
echo " Phase 1: Clean Accuracy vs SNR (AWGN)"
echo "=============================================="
python evaluate.py \
  --model_path ${RESULTS_DIR}/baseline_cnn/best_model.pth \
  --data_path $DATA_PATH \
  --dataset_version $DATASET_VERSION \
  --num_classes $NUM_CLASSES \
  --input_length $INPUT_LENGTH \
  --eval_mode clean_snr \
  --output_dir ${RESULTS_DIR}/baseline_cnn \
  --device $DEVICE \
  --seed $SEED \
  --num_trials 10

echo ""
echo "=============================================="
echo " Phase 2: PGD Attack vs SNR (AWGN)"
echo "=============================================="
python evaluate.py \
  --model_path ${RESULTS_DIR}/baseline_cnn/best_model.pth \
  --data_path $DATA_PATH \
  --dataset_version $DATASET_VERSION \
  --num_classes $NUM_CLASSES \
  --input_length $INPUT_LENGTH \
  --eval_mode attack_snr \
  --attack_type pgd \
  --channel_type awgn \
  --rho 0.01 \
  --pgd_steps 10 \
  --output_dir ${RESULTS_DIR}/baseline_cnn \
  --device $DEVICE \
  --seed $SEED \
  --num_trials 10

echo ""
echo "=============================================="
echo " Phase 2: FGSM Attack vs SNR (AWGN)"
echo "=============================================="
python evaluate.py \
  --model_path ${RESULTS_DIR}/baseline_cnn/best_model.pth \
  --data_path $DATA_PATH \
  --dataset_version $DATASET_VERSION \
  --num_classes $NUM_CLASSES \
  --input_length $INPUT_LENGTH \
  --eval_mode attack_snr \
  --attack_type fgsm \
  --channel_type awgn \
  --rho 0.01 \
  --output_dir ${RESULTS_DIR}/baseline_cnn \
  --device $DEVICE \
  --seed $SEED \
  --num_trials 10

echo ""
echo "=============================================="
echo " Phase 2: Attack vs Perturbation Budget"
echo "=============================================="
python evaluate.py \
  --model_path ${RESULTS_DIR}/baseline_cnn/best_model.pth \
  --data_path $DATA_PATH \
  --dataset_version $DATASET_VERSION \
  --num_classes $NUM_CLASSES \
  --input_length $INPUT_LENGTH \
  --eval_mode attack_budget \
  --attack_type pgd \
  --channel_type awgn \
  --pgd_steps 10 \
  --output_dir ${RESULTS_DIR}/baseline_cnn \
  --device $DEVICE \
  --seed $SEED \
  --num_trials 10

echo ""
echo "=============================================="
echo " Phase 3: Defense Training & Evaluation (2018)"
echo "=============================================="
python train_defenses.py \
  --data_path $DATA_PATH \
  --dataset_version $DATASET_VERSION \
  --num_classes $NUM_CLASSES \
  --input_length $INPUT_LENGTH \
  --epochs 60 \
  --batch_size 256 \
  --lr 0.001 \
  --seed $SEED \
  --rho 0.01 \
  --pgd_steps 5 \
  --sigma 0.01 \
  --output_dir $RESULTS_DIR \
  --eval_snr 0 10 \
  --device $DEVICE

echo ""
echo "=============================================="
echo " Phase 4: Per-Class Vulnerability (2018)"
echo "=============================================="
python eval_per_class.py \
  --model_path ${RESULTS_DIR}/baseline_cnn/best_model.pth \
  --data_path $DATA_PATH \
  --dataset_version $DATASET_VERSION \
  --num_classes $NUM_CLASSES \
  --input_length $INPUT_LENGTH \
  --snr 0 10 18 \
  --rho 0.01 \
  --attack pgd \
  --pgd_steps 10 \
  --num_trials 10 \
  --output_dir $RESULTS_DIR \
  --device $DEVICE

echo ""
echo "=============================================="
echo " Generate Figures (2018)"
echo "=============================================="
python plot_results.py \
  --results_dir $RESULTS_DIR \
  --output_dir experiments/figures_2018

echo ""
echo "=============================================="
echo " ALL 2018 EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Results: ${RESULTS_DIR}/"
echo "Figures: experiments/figures_2018/"
