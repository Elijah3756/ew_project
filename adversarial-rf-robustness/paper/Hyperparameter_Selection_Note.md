# Hyperparameter selection — text for your paper

Use the **short paragraph** in Section 4 (Experimental Setup / Training) if you ran `tune_train.py` and adopted the winning settings. If you did **not** run a search, keep your original fixed-hyperparameter wording (or say values were chosen from common practice / pilot runs).

---

## Short paragraph (paste-ready)

**Hyperparameter selection.** Learning rate, mini-batch size, and number of training epochs were selected by an exhaustive grid search over candidate values (Cartesian product), holding the train/validation/test split and random seed fixed. For each configuration, we trained the baseline CNN from scratch and recorded validation accuracy at every epoch; the configuration with the **maximum validation accuracy** (equivalently, the best checkpoint on the validation set) was retained. The same Adam optimizer and cosine learning-rate schedule were used within each run; weight decay was held fixed across the grid. After selecting hyperparameters, we **retrained** the final reported model with those settings and performed the full evaluation protocol (including per-SNR test evaluation). Test data were not used for hyperparameter selection.

---

## Slightly longer (if reviewers want more detail)

**Hyperparameter selection.** We tuned three training hyperparameters—the initial learning rate (Adam), mini-batch size, and total number of epochs—using a **grid search**: all combinations of prespecified candidate values were evaluated. Each trial used an identical random seed and the same 70/15/15 train/validation/test split (Section 4.1). Within each trial, we optimized the cross-entropy loss with Adam, applied cosine annealing of the learning rate over the chosen epoch budget, and saved the weights achieving the highest **validation** classification accuracy. The grid point with the highest peak validation accuracy was chosen. Final results in the paper use that configuration; the held-out **test** set was used only for reporting generalization and robustness metrics after hyperparameters were fixed, not for tuning.

---

## One sentence (minimal)

Hyperparameters (learning rate, batch size, number of epochs) were chosen by grid search to maximize validation accuracy on the RadioML split; the test set was reserved for final evaluation only.

---

## Reproducibility / code (optional footnote)

Hyperparameter search is implemented in `tune_train.py` in the accompanying code repository; each trial invokes `train.py` with the same data pipeline and records per-epoch validation metrics in `training_history.csv`.

---

## If you update §4.3 numerically

After you run tuning, replace the fixed numbers in **§4.3 Training Configuration** (learning rate, batch size, epochs) with your **selected** values and add a cross-reference, e.g.:

> *These values were obtained from the grid search described above (maximum validation accuracy).*

If the paper still describes 0.001 / 256 / 60 without a search, align the text with what you actually did to avoid inconsistency.
