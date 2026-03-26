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

## Staged Tuning Workflow

Run the coarse-to-fine tuning workflow when you want the baseline and defense
hyperparameters selected from data instead of using the fixed notebook defaults.
The workflow requires both the 2016 and 2018 datasets.

```bash
python run_tuning_workflow.py \
  --data_path_2016 /path/to/RML2016.10a_dict.pkl \
  --data_path_2018 /path/to/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --output_root experiments/tuning_workflow \
  --device auto
```

Key outputs:

- `experiments/tuning_workflow/workflow_summary.csv`
- `experiments/tuning_workflow/baseline/top_<search_name>.json`
- `experiments/tuning_workflow/baseline/refine_candidates_<search_name>.json`
- `experiments/tuning_workflow/defenses/defense_tuning/best_by_defense_defense_tuning.json`

For a fast validation run, use the same entrypoint with a reduced grid. The
Colab notebook includes an example smoke-test command that constrains both the
coarse search and the refinement expansion.

During tuning, baseline trials call `train.py --skip_snr_sweep` so the search
optimizes on validation accuracy without paying for the full post-training SNR
sweep on every candidate.

## Key Parameters

- **SNR Range:** -10 to 20 dB
- **Attacks:** FGSM, PGD
- **Perturbation Budgets (rho):** 0.5%, 1%, 2%, 5%
- **Channels:** AWGN, Rayleigh fading, CFO
- **Defenses:** Channel augmentation, adversarial training, noise injection/smoothing
