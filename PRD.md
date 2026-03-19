# Product Requirements Document (PRD)

## Project Title
Adversarial Robustness of AI-Based RF Signal Classification Under Channel Impairments

## 1. Purpose

Develop and evaluate adversarial attacks and defenses for deep learning-based RF modulation classifiers operating in contested electromagnetic environments. The goal is to quantify robustness margins under realistic channel impairments and identify computationally feasible defense strategies.

---

## 2. Background

AI-driven RF classification enables machine-speed spectrum awareness and adaptive electromagnetic operations. However, deep learning models are vulnerable to adversarial perturbations. This project evaluates waveform-level adversarial robustness under realistic channel conditions (AWGN, Rayleigh fading, CFO/timing offset).

---

## 3. Objectives

1. Define a physically plausible adversarial threat model for I/Q-based classifiers.
2. Quantify attack success rates across SNR and channel conditions.
3. Compare robustness improvements from:
   - Channel augmentation
   - Adversarial training
   - Noise injection / randomized smoothing
4. Measure robustness-compute tradeoffs (GPU hours, latency, model size).

---

## 4. Scope

### In Scope

- Modulation classification (I/Q input windows)
- CNN baseline (mandatory)
- Optional second model (CLDNN or small Transformer)
- Attacks: FGSM, PGD
- Channels:
  - AWGN
  - Rayleigh fading
  - CFO (or timing offset)
- Evaluation across SNR range (-10 to 20 dB)

### Out of Scope

- Over-the-air hardware experiments
- Classified datasets
- Large-scale emitter identification
- Wideband detection tasks

---

## 5. Technical Requirements

### 5.1 Dataset

- Public modulation classification dataset (e.g., RadioML-style)
- Fixed train/validation/test splits
- Deterministic seeds

### 5.2 Model Requirements

- PyTorch implementation
- GPU acceleration
- Mixed precision training (if beneficial)
- Model logging and checkpointing

### 5.3 Channel Model Requirements

- Differentiable channel layers:
  - AWGN
  - Rayleigh fading
  - CFO or timing offset
- Configurable SNR sweep

### 5.4 Adversarial Attack Requirements

- FGSM
- PGD (multi-step)
- Power-constrained perturbation:
  - ||delta||_2 / ||x||_2 <= rho
  - rho in {0.5%, 1%, 2%, 5%}

### 5.5 Defense Requirements

- Adversarial training (PGD-lite)
- Noise injection / smoothing
- Channel augmentation training

---

## 6. Evaluation Metrics

### Primary Metrics

- Clean Accuracy vs SNR
- Robust Accuracy vs SNR
- Attack Success Rate vs perturbation budget
- Robustness margin under Rayleigh fading

### Secondary Metrics

- Training time (GPU hours)
- Inference latency (ms/sample)
- Parameter count

---

## 7. Deliverables

1. Reproducible PyTorch repository
2. Experimental results:
   - Baseline performance
   - Attack performance
   - Defense comparison
3. Publication-ready figures:
   - Accuracy vs SNR
   - Robust accuracy vs SNR
   - Attack success vs perturbation budget
4. Submission-ready manuscript

---

## 8. Success Criteria

- Demonstrate measurable degradation under adversarial perturbation.
- Show at least one defense improves robustness without catastrophic clean accuracy loss.
- Produce statistically consistent results across random seeds.
- Submit manuscript by April 15, 2026.

---

## 9. Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Dataset quality limitations | Use synthetic augmentation |
| Excess experiment scope | Limit to 1 dataset, 1-2 models |
| Compute overload | Prioritize baseline + PGD only |
| Weak novelty | Emphasize channel-aware adversarial framing |

---

## 10. Future Extensions (Optional)

- Transferability analysis across models
- Universal perturbations
- Certified robustness bounds
- Multi-agent adversarial modeling
