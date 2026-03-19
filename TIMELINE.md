# Project Timeline

**Submission Deadline:** April 15, 2026
**Start Date:** February 22, 2026
**Duration:** ~7.5 weeks

---

## Phase 0 -- Setup (Feb 22-Feb 27)

- Define threat model and experimental matrix
- Select dataset
- Build repo structure
- Implement data loader
- Train initial CNN baseline (clean)

**Deliverable:**
- Clean accuracy vs SNR plot
- Stable training pipeline

---

## Phase 1 -- Channel Modeling (Feb 28-Mar 6)

- Implement AWGN layer
- Implement Rayleigh fading
- Implement CFO (or timing offset)
- Train channel-augmented model
- Evaluate under SNR sweep

**Deliverable:**
- Clean vs channel-aug accuracy comparison figure

---

## Phase 2 -- Adversarial Attacks (Mar 7-Mar 15)

- Implement FGSM
- Implement PGD (multi-step)
- Validate perturbation constraint
- Run attacks at selected SNR levels
- Expand to full SNR sweep

**Deliverable:**
- Robust accuracy vs SNR
- Attack success vs perturbation budget

---

## Phase 3 -- Defense Mechanisms (Mar 16-Mar 24)

- Implement adversarial training
- Implement noise injection / smoothing
- Compare clean vs robust accuracy
- Record compute overhead

**Deliverable:**
- Defense comparison figure
- Robustness-compute tradeoff table

---

## Phase 4 -- Optional Extensions (Mar 25-Apr 2)

If time allows:
- Add second model (CLDNN or Transformer)
- Transferability experiments
- Perturbation budget ablation study

**Deliverable:**
- Transferability matrix (optional)
- Additional robustness plots

---

## Phase 5 -- Writing & Finalization (Apr 3-Apr 15)

**Apr 3-Apr 9:**
- Write Methods + Experimental Setup
- Draft Results and Discussion
- Insert final figures

**Apr 10-Apr 13:**
- Write Introduction + Related Work
- Refine abstract

**Apr 14-Apr 15:**
- Final proofread
- Re-run key experiments (fixed seeds)
- Submit manuscript

---

## Milestone Summary

| Date   | Milestone                    |
|--------|------------------------------|
| Feb 27 | Baseline model complete      |
| Mar 6  | Channel modeling complete    |
| Mar 15 | Attack evaluation complete   |
| Mar 24 | Defense experiments complete |
| Apr 9  | Full draft complete          |
| Apr 15 | Submission                   |

---

## Minimum Viable Scope (If Time-Constrained)

- 1 dataset
- 1 CNN model
- FGSM + PGD
- AWGN + Rayleigh
- Adversarial training defense
- Robust accuracy vs SNR as primary result
