# Adversarial Robustness of AI-Based RF Signal Classification

Physical-layer adversarial robustness evaluation for deep learning-based RF modulation classifiers under realistic channel impairments.

**Target:** CDR 2026 EW Special Issue
**Deadline:** April 15, 2026

## Repository Structure

```
adversarial-rf-robustness/
|-- data/                  # Dataset loading and preprocessing
|   |-- raw/               # Raw dataset files (not tracked in git)
|   |-- processed/         # Preprocessed splits
|-- models/                # Model architectures (CNN, optional CLDNN/Transformer)
|-- channels/              # Differentiable channel layers (AWGN, Rayleigh, CFO)
|-- attacks/               # Adversarial attack implementations (FGSM, PGD)
|-- defenses/              # Defense mechanisms (adv training, noise injection)
|-- experiments/           # Experiment runners and configs
|   |-- configs/           # YAML experiment configurations
|   |-- results/           # CSV metric logs
|   |-- figures/           # Generated plots
|-- paper/                 # Manuscript source
|   |-- sections/          # Per-section drafts
|   |-- figures/           # Publication-ready figures
|-- utils/                 # Shared utilities (metrics, logging, visualization)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline CNN
python train.py --config experiments/configs/baseline_cnn.yaml

# Evaluate under attacks
python evaluate.py --config experiments/configs/attack_eval.yaml
```

## Key Parameters

- **SNR Range:** -10 to 20 dB
- **Attacks:** FGSM, PGD
- **Perturbation Budgets (rho):** 0.5%, 1%, 2%, 5%
- **Channels:** AWGN, Rayleigh fading, CFO
- **Defenses:** Channel augmentation, adversarial training, noise injection/smoothing
