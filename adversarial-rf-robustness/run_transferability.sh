#!/bin/bash
# =============================================================
# Transferability analysis: adversarial examples from baseline
# tested against all defense models (2016 dataset)
# Usage: bash run_transferability.sh
# =============================================================

set -e

DATA_PATH="data/raw/RML2016.10a_dict.pkl"
DEVICE="auto"
RESULTS_DIR="experiments/results"

echo "=============================================="
echo " Adversarial Transferability Analysis"
echo "=============================================="

# Source: baseline CNN, Targets: all defense models
python eval_transferability.py \
  --source_path ${RESULTS_DIR}/baseline_cnn/best_model.pth \
  --source_name baseline \
  --target_path \
    ${RESULTS_DIR}/channel_aug/best_model.pth \
    ${RESULTS_DIR}/adv_train/best_model.pth \
    ${RESULTS_DIR}/noise_inject/best_model.pth \
    ${RESULTS_DIR}/adv_train_channel/best_model.pth \
  --target_names channel_aug adv_train noise_inject adv_train_channel \
  --data_path $DATA_PATH \
  --dataset_version 2016.10a \
  --num_classes 11 \
  --input_length 128 \
  --snr 0 10 \
  --rho 0.01 \
  --attack pgd \
  --pgd_steps 10 \
  --num_trials 10 \
  --output_dir $RESULTS_DIR \
  --device $DEVICE

# =============================================================
# 2018.01a Transferability Analysis
# =============================================================
RESULTS_2018="experiments/results_2018"
DATA_2018="data/raw/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"

echo ""
echo "=============================================="
echo " Adversarial Transferability Analysis (2018)"
echo "=============================================="

python eval_transferability.py \
  --source_path ${RESULTS_2018}/baseline_cnn/best_model.pth \
  --source_name baseline \
  --target_path \
    ${RESULTS_2018}/channel_aug/best_model.pth \
    ${RESULTS_2018}/adv_train/best_model.pth \
    ${RESULTS_2018}/noise_inject/best_model.pth \
    ${RESULTS_2018}/adv_train_channel/best_model.pth \
  --target_names "Channel Aug." "Adv. Training" "Noise Inj." "Adv.+Chan. Aug." \
  --data_path $DATA_2018 \
  --dataset_version 2018.01a \
  --num_classes 24 \
  --input_length 1024 \
  --snr 0 10 \
  --rho 0.01 \
  --attack pgd \
  --pgd_steps 10 \
  --num_trials 10 \
  --output_dir $RESULTS_2018 \
  --device $DEVICE

echo ""
echo "=============================================="
echo " TRANSFERABILITY ANALYSIS COMPLETE"
echo "=============================================="
echo "Results (2016): ${RESULTS_DIR}/transferability_analysis.csv"
echo "Results (2018): ${RESULTS_2018}/transferability_analysis.csv"
